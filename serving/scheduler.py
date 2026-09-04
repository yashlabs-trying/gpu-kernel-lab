"""Decode-first continuous batching with bounded, round-robin chunked prefill."""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass,field
from enum import Enum
from .sampling import SamplingParams


class RequestState(str,Enum):
    WAITING='waiting'; PREFILL='prefill'; DECODE='decode'; FINISHED='finished'; CANCELLED='cancelled'; FAILED='failed'


@dataclass
class Request:
    request_id:str
    prompt_token_ids:tuple[int,...]
    max_new_tokens:int
    sampling:SamplingParams=field(default_factory=SamplingParams)
    state:RequestState=RequestState.WAITING
    prefill_cursor:int=0
    generated:list[int]=field(default_factory=list)
    error:str|None=None

    @property
    def remaining_prompt(self): return len(self.prompt_token_ids)-self.prefill_cursor
    @property
    def finished(self): return self.state in (RequestState.FINISHED,RequestState.CANCELLED,RequestState.FAILED)


@dataclass(frozen=True)
class PrefillWork:
    request_id:str
    token_ids:tuple[int,...]
    start_position:int


@dataclass(frozen=True)
class StepPlan:
    decode_ids:tuple[str,...]
    prefill:tuple[PrefillWork,...]
    graph_batch_size:int|None


class ContinuousBatchScheduler:
    """Control-plane scheduler; an executor supplies model-specific GPU calls."""
    def __init__(self,allocator,graph_buckets,*,max_batch_size=16,prefill_chunk_size=256,max_prefill_tokens=512):
        if min(max_batch_size,prefill_chunk_size,max_prefill_tokens)<=0: raise ValueError('limits must be positive')
        self.allocator=allocator; self.graph_buckets=graph_buckets
        self.max_batch_size=max_batch_size; self.prefill_chunk_size=prefill_chunk_size; self.max_prefill_tokens=max_prefill_tokens
        self.requests:dict[str,Request]={}; self._prefill=deque(); self._decode=deque()

    def submit(self,request):
        if not request.request_id or request.request_id in self.requests: raise ValueError('request id must be unique')
        if not request.prompt_token_ids or request.max_new_tokens<=0: raise ValueError('nonempty prompt and positive max_new_tokens required')
        self.allocator.create(request.request_id); request.state=RequestState.PREFILL
        self.requests[request.request_id]=request; self._prefill.append(request.request_id)

    def cancel(self,request_id):
        request=self.requests[request_id]
        if request.finished: return False
        request.state=RequestState.CANCELLED; self.allocator.release(request_id); return True

    def plan(self):
        decode=[]
        for _ in range(min(len(self._decode),self.max_batch_size)):
            request_id=self._decode.popleft(); request=self.requests[request_id]
            if request.state==RequestState.DECODE:
                decode.append(request_id); self._decode.append(request_id)
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
            prefill.append(PrefillWork(request_id,request.prompt_token_ids[start:start+count],start))
            token_budget-=count; self._prefill.append(request_id)
        bucket=self.graph_buckets.select(len(decode),max((self.allocator.length(i) for i in decode),default=0)) if decode else None
        return StepPlan(tuple(decode),tuple(prefill),None if bucket is None else bucket[0])

    def complete_prefill(self,request_id,count):
        request=self.requests[request_id]
        if request.state!=RequestState.PREFILL or count<=0 or count>request.remaining_prompt: raise ValueError('invalid prefill completion')
        self.allocator.append_slots(request_id,count); request.prefill_cursor+=count
        if request.remaining_prompt==0:
            request.state=RequestState.DECODE
            self._remove(self._prefill,request_id); self._decode.append(request_id)

    def complete_decode(self,request_id,token_id,stopped=False):
        request=self.requests[request_id]
        if request.state!=RequestState.DECODE: raise ValueError('request is not decoding')
        self.allocator.append_slots(request_id,1); request.generated.append(int(token_id))
        if stopped or len(request.generated)>=request.max_new_tokens:
            request.state=RequestState.FINISHED; self._remove(self._decode,request_id); self.allocator.release(request_id)

    def fail(self,request_id,error):
        request=self.requests[request_id]; request.state=RequestState.FAILED; request.error=str(error)
        self._remove(self._prefill,request_id); self._remove(self._decode,request_id); self.allocator.release(request_id)

    @staticmethod
    def _remove(queue,value):
        try: queue.remove(value)
        except ValueError: pass

