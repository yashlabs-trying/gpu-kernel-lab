"""Asynchronous serving loop joining scheduler, executor, sampler, and events."""
from __future__ import annotations
import asyncio
from dataclasses import dataclass
import json
import logging
import time
from typing import Protocol

from .metrics import ServingMetrics
from .sampling import Sampler
from .scheduler import AdmissionError,ContinuousBatchScheduler,Request,RequestState


class Executor(Protocol):
    async def prefill(self,work,allocator): ...
    async def decode(self,work,metadata,graph_batch_size): ...  # request_id -> logits


@dataclass(frozen=True)
class TokenEvent:
    request_id:str; token_id:int|None=None; logprob:float|None=None
    top_logprobs:tuple=(); finished:bool=False; finish_reason:str|None=None; error:str|None=None


class JsonFormatter(logging.Formatter):
    def format(self,record):
        return json.dumps({'time':time.time(),'level':record.levelname,'message':record.getMessage(),
            **getattr(record,'fields',{})},separators=(',',':'))


class ServingEngine:
    def __init__(self,scheduler:ContinuousBatchScheduler,executor:Executor,*,idle_sleep_s=.001,logger=None):
        self.scheduler=scheduler; self.executor=executor; self.idle_sleep_s=idle_sleep_s
        self.metrics=ServingMetrics(); self.events={}; self.samplers={}; self.first_token_at={}; self.last_token_at={}
        self._lock=asyncio.Lock(); self._task=None; self._stopping=False
        self.log=logger or logging.getLogger('kernellab.serving')

    async def start(self):
        if self._task is None: self._task=asyncio.create_task(self._run(),name='kernellab-serving-loop')

    async def close(self):
        self._stopping=True
        if self._task is not None: await self._task

    async def submit(self,request):
        queue=asyncio.Queue()
        async with self._lock:
            try: self.scheduler.submit(request)
            except AdmissionError:
                self.metrics.rejected+=1; raise
            self.events[request.request_id]=queue; self.samplers[request.request_id]=Sampler(request.sampling)
            self.metrics.submitted+=1; self.metrics.prompt_tokens+=len(request.prompt_token_ids)
        self._log('submitted',request_id=request.request_id,priority=request.priority,prompt_tokens=len(request.prompt_token_ids))
        return queue

    async def cancel(self,request_id):
        async with self._lock:
            changed=self.scheduler.cancel(request_id)
            if changed:
                self.metrics.cancelled+=1; await self._finish_event(request_id,'cancelled')
        return changed

    async def _run(self):
        while not self._stopping or self.active_requests:
            worked=await self.step()
            if not worked: await asyncio.sleep(self.idle_sleep_s)

    async def step(self):
        async with self._lock:
            expired=self.scheduler.expire()
            for request_id in expired:
                self.metrics.timed_out+=1; await self._finish_event(request_id,'timeout')
            plan=self.scheduler.plan()
        if not plan.decode and not plan.prefill: return False
        try:
            if plan.prefill:
                await self.executor.prefill(plan.prefill,self.scheduler.allocator)
                async with self._lock:
                    for work in plan.prefill: self.scheduler.complete_prefill(work.request_id,len(work.token_ids))
            if plan.decode:
                metadata=self.scheduler.allocator.metadata(tuple(x.request_id for x in plan.decode))
                logits=await self.executor.decode(plan.decode,metadata,plan.graph_batch_size)
                async with self._lock:
                    for work in plan.decode:
                        request=self.scheduler.requests[work.request_id]; now=time.monotonic()
                        result=self.samplers[work.request_id].sample(logits[work.request_id],request.prompt_token_ids+tuple(request.generated))
                        self.scheduler.complete_decode(work.request_id,result.token_id,result.stopped)
                        self.metrics.generated_tokens+=1
                        if work.request_id not in self.first_token_at:
                            self.first_token_at[work.request_id]=now
                            self.metrics.observe(self.metrics.ttft_ms,(now-request.submitted_at)*1000)
                        else: self.metrics.observe(self.metrics.itl_ms,(now-self.last_token_at[work.request_id])*1000)
                        self.last_token_at[work.request_id]=now
                        await self.events[work.request_id].put(TokenEvent(work.request_id,result.token_id,result.logprob,result.top_logprobs))
                        if request.finished:
                            self.metrics.completed+=1; await self._finish_event(work.request_id,'stop' if result.stopped else 'length')
        except Exception as error:
            async with self._lock:
                for request_id in {x.request_id for x in plan.decode}|{x.request_id for x in plan.prefill}:
                    if not self.scheduler.requests[request_id].finished:
                        self.scheduler.fail(request_id,error); self.metrics.failed+=1
                        await self._finish_event(request_id,'error',str(error))
            self.log.exception('serving step failed')
        return True

    async def _finish_event(self,request_id,reason,error=None):
        queue=self.events.get(request_id)
        if queue is not None: await queue.put(TokenEvent(request_id,finished=True,finish_reason=reason,error=error))
        self._log('finished',request_id=request_id,reason=reason,error=error)

    @property
    def active_requests(self): return sum(not r.finished for r in self.scheduler.requests.values())
    def snapshot(self):
        result=self.metrics.snapshot(self.scheduler.allocator,self.active_requests)
        result['kv']['reserved_blocks']=self.scheduler.admitted_blocks
        return result
    def _log(self,message,**fields): self.log.info(message,extra={'fields':fields})
