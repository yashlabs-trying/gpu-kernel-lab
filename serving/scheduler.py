"""Decode-first continuous batching with bounded, round-robin chunked prefill."""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass,field
from enum import Enum
import time
from .sampling import SamplingParams


class RequestState(str,Enum):
    WAITING='waiting'; PREFILL='prefill'; DECODE='decode'; FINISHED='finished'; CANCELLED='cancelled'; TIMED_OUT='timed_out'; REJECTED='rejected'; FAILED='failed'


class AdmissionError(RuntimeError): pass


@dataclass
class Request:
    request_id:str
    prompt_token_ids:tuple[int,...]
    max_new_tokens:int
    sampling:SamplingParams=field(default_factory=SamplingParams)
    priority:int=0
    timeout_s:float|None=60.0
    state:RequestState=RequestState.WAITING
    prefill_cursor:int=0
    generated:list[int]=field(default_factory=list)
    error:str|None=None
    submitted_at:float=field(default_factory=time.monotonic)

    @property
    def remaining_prompt(self): return len(self.prompt_token_ids)-self.prefill_cursor
    @property
    def finished(self): return self.state in (RequestState.FINISHED,RequestState.CANCELLED,RequestState.TIMED_OUT,RequestState.REJECTED,RequestState.FAILED)


@dataclass(frozen=True)
class PrefillWork:
    request_id:str
    token_ids:tuple[int,...]
    start_position:int
    slots:tuple[tuple[int,int],...]


@dataclass(frozen=True)
class DecodeWork:
    request_id:str
    position:int
    slot:tuple[int,int]


@dataclass(frozen=True)
class StepPlan:
    decode:tuple[DecodeWork,...]
    prefill:tuple[PrefillWork,...]
    graph_batch_size:int|None


class ContinuousBatchScheduler:
    """Control-plane scheduler; an executor supplies model-specific GPU calls."""
    def __init__(self,allocator,graph_buckets,*,max_batch_size=16,prefill_chunk_size=256,max_prefill_tokens=512):
        if min(max_batch_size,prefill_chunk_size,max_prefill_tokens)<=0: raise ValueError('limits must be positive')
        self.allocator=allocator; self.graph_buckets=graph_buckets
        self.max_batch_size=max_batch_size; self.prefill_chunk_size=prefill_chunk_size; self.max_prefill_tokens=max_prefill_tokens
        self.requests:dict[str,Request]={}; self._prefill=deque(); self._decode=deque()
        self._inflight:dict[str,tuple[str,int]]={}
        self._reserved_blocks:dict[str,int]={}

    def submit(self,request):
        if not request.request_id or request.request_id in self.requests: raise ValueError('request id must be unique')
        if not request.prompt_token_ids or request.max_new_tokens<=0: raise ValueError('nonempty prompt and positive max_new_tokens required')
        if request.timeout_s is not None and request.timeout_s<=0: raise ValueError('timeout must be positive')
        reserve=self.allocator.blocks_for_tokens(len(request.prompt_token_ids)+request.max_new_tokens)
        if reserve>self.allocator.num_blocks-sum(self._reserved_blocks.values()):
            request.state=RequestState.REJECTED; request.error='insufficient KV capacity'; self.requests[request.request_id]=request
            raise AdmissionError(request.error)
        self.allocator.create(request.request_id); request.state=RequestState.PREFILL
        request.submitted_at=time.monotonic(); self.requests[request.request_id]=request
        self._reserved_blocks[request.request_id]=reserve; self._prefill.append(request.request_id)

    def cancel(self,request_id):
        request=self.requests[request_id]
        if request.finished: return False
        request.state=RequestState.CANCELLED; self._cleanup(request_id); return True

    def plan(self):
        if self._inflight: raise RuntimeError('complete or fail the current plan before planning again')
        self.expire()
        self._decode=deque(sorted(self._decode,key=lambda i:-self.requests[i].priority))
        self._prefill=deque(sorted(self._prefill,key=lambda i:-self.requests[i].priority))
        decode=[]
        for _ in range(min(len(self._decode),self.max_batch_size)):
            request_id=self._decode.popleft(); request=self.requests[request_id]
            if request.state==RequestState.DECODE:
                position=self.allocator.length(request_id)
                slot=self.allocator.append_slots(request_id,1)[0]
                decode.append(DecodeWork(request_id,position,slot)); self._inflight[request_id]=('decode',1)
                self._decode.append(request_id)
        remaining_slots=self.max_batch_size-len(decode); token_budget=self.max_prefill_tokens; prefill=[]
        # One chunk per request per iteration prevents a long prompt monopolizing prefill.
        visits=min(len(self._prefill),remaining_slots)
        for _ in range(visits):
            request_id=self._prefill.popleft(); request=self.requests[request_id]
            if request.state!=RequestState.PREFILL: continue
            count=min(request.remaining_prompt,self.prefill_chunk_size,token_budget)
            if count<=0:
                self._prefill.appendleft(request_id); break
            start=request.prefill_cursor
            slots=tuple(self.allocator.append_slots(request_id,count))
            prefill.append(PrefillWork(request_id,request.prompt_token_ids[start:start+count],start,slots))
            self._inflight[request_id]=('prefill',count)
            token_budget-=count; self._prefill.append(request_id)
        bucket=self.graph_buckets.select(len(decode),max((self.allocator.length(x.request_id) for x in decode),default=0)) if decode else None
        return StepPlan(tuple(decode),tuple(prefill),None if bucket is None else bucket[0])

    def complete_prefill(self,request_id,count):
        request=self.requests[request_id]
        if self._inflight.get(request_id)!=('prefill',count) or request.state!=RequestState.PREFILL or count<=0 or count>request.remaining_prompt: raise ValueError('invalid prefill completion')
        self._inflight.pop(request_id)
        request.prefill_cursor+=count
        if request.remaining_prompt==0:
            request.state=RequestState.DECODE
            self._remove(self._prefill,request_id); self._decode.append(request_id)

    def complete_decode(self,request_id,token_id,stopped=False):
        request=self.requests[request_id]
        if self._inflight.get(request_id)!=('decode',1) or request.state!=RequestState.DECODE: raise ValueError('request is not decoding')
        self._inflight.pop(request_id)
        request.generated.append(int(token_id))
        if stopped or len(request.generated)>=request.max_new_tokens:
            request.state=RequestState.FINISHED; self._cleanup(request_id)

    def fail(self,request_id,error):
        request=self.requests[request_id]; request.state=RequestState.FAILED; request.error=str(error)
        self._cleanup(request_id)

    def expire(self,now=None):
        now=time.monotonic() if now is None else now; expired=[]
        for request_id,request in tuple(self.requests.items()):
            if not request.finished and request.timeout_s is not None and now-request.submitted_at>=request.timeout_s:
                request.state=RequestState.TIMED_OUT; request.error='request deadline exceeded'; self._cleanup(request_id); expired.append(request_id)
        return tuple(expired)

    @property
    def admitted_blocks(self): return sum(self._reserved_blocks.values())

    def _cleanup(self,request_id):
        self._inflight.pop(request_id,None); self._reserved_blocks.pop(request_id,None)
        self._remove(self._prefill,request_id); self._remove(self._decode,request_id); self.allocator.release(request_id)

    @staticmethod
    def _remove(queue,value):
        try: queue.remove(value)
        except ValueError: pass
