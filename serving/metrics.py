"""Bounded in-process metrics for latency, throughput, queueing, and KV memory."""
from collections import deque
from dataclasses import dataclass,field
import math
import time


def percentile(values,p):
    if not values: return None
    ordered=sorted(values); position=(len(ordered)-1)*p/100
    low=math.floor(position); high=math.ceil(position)
    return ordered[low] if low==high else ordered[low]+(ordered[high]-ordered[low])*(position-low)


@dataclass
class ServingMetrics:
    window:int=10000
    started_at:float=field(default_factory=time.monotonic)
    submitted:int=0; completed:int=0; cancelled:int=0; timed_out:int=0; rejected:int=0; failed:int=0
    prompt_tokens:int=0; generated_tokens:int=0
    ttft_ms:deque=field(default_factory=deque); itl_ms:deque=field(default_factory=deque)

    def observe(self,target,value):
        target.append(float(value))
        while len(target)>self.window: target.popleft()

    def snapshot(self,allocator,active):
        elapsed=max(time.monotonic()-self.started_at,1e-9)
        def summary(values): return {f'p{p}':percentile(values,p) for p in (50,90,95,99)}
        return {'uptime_s':elapsed,'requests':{'submitted':self.submitted,'completed':self.completed,
            'cancelled':self.cancelled,'timed_out':self.timed_out,'rejected':self.rejected,'failed':self.failed,'active':active},
            'tokens':{'prompt':self.prompt_tokens,'generated':self.generated_tokens,
                'generated_per_second':self.generated_tokens/elapsed},
            'latency_ms':{'ttft':summary(self.ttft_ms),'itl':summary(self.itl_ms)},
            'kv':{'total_blocks':allocator.num_blocks,'free_blocks':allocator.free_blocks,
                'used_blocks':allocator.num_blocks-allocator.free_blocks,'block_size':allocator.block_size,
                'storage_bytes':allocator.storage_bytes}}
