"""Fixed-block K/V storage with deterministic allocation and immediate reuse."""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from threading import RLock

import torch


@dataclass(frozen=True)
class BatchKVMetadata:
    request_ids: tuple[str,...]
    block_tables: torch.Tensor  # int32 [B,max_blocks], -1 means unused
    lengths: torch.Tensor       # int32 [B]
    query_positions: torch.Tensor  # int64 [B]


@dataclass
class _Allocation:
    blocks: list[int]
    length: int=0


class PagedKVAllocator:
    """Own `[layer,block,kv_head,block_token,head_dim]` K/V tensors.

    Allocation metadata is CPU-side and protected by a lock. GPU kernels consume
    compact block tables and lengths generated once per scheduled batch.
    """
    def __init__(self,*,num_layers,num_blocks,block_size,num_kv_heads,head_dim,
                 dtype=torch.bfloat16,device='cuda'):
        dimensions=(num_layers,num_blocks,block_size,num_kv_heads,head_dim)
        if any(not isinstance(x,int) or x<=0 for x in dimensions):
            raise ValueError('all dimensions must be positive integers')
        if block_size & (block_size-1):
            raise ValueError('block_size must be a power of two')
        self.num_layers,self.num_blocks,self.block_size,self.num_kv_heads,self.head_dim=dimensions
        shape=(num_layers,num_blocks,num_kv_heads,block_size,head_dim)
        self.keys=torch.empty(shape,dtype=dtype,device=device)
        self.values=torch.empty(shape,dtype=dtype,device=device)
        self._free=list(range(num_blocks)); heapq.heapify(self._free)
        self._requests:dict[str,_Allocation]={}
        self._lock=RLock()

    def create(self,request_id):
        with self._lock:
            if not request_id or request_id in self._requests:
                raise ValueError('request id must be nonempty and unique')
            self._requests[request_id]=_Allocation([])

    def append_slots(self,request_id,count):
        """Reserve logical positions atomically and return `(block,offset)` pairs."""
        if not isinstance(count,int) or count<0: raise ValueError('count must be nonnegative')
        with self._lock:
            allocation=self._get(request_id)
            final=allocation.length+count
            required=math.ceil(final/self.block_size)
            additional=required-len(allocation.blocks)
            if additional>len(self._free):
                raise MemoryError(f'KV blocks exhausted: need {additional}, free {len(self._free)}')
            allocation.blocks.extend(heapq.heappop(self._free) for _ in range(additional))
            slots=[]
            for position in range(allocation.length,final):
                slots.append((allocation.blocks[position//self.block_size],position&(self.block_size-1)))
            allocation.length=final
            return slots

    def release(self,request_id):
        with self._lock:
            allocation=self._requests.pop(request_id,None)
            if allocation is None: return False
            for block in allocation.blocks: heapq.heappush(self._free,block)
            return True

    def metadata(self,request_ids,device=None):
        with self._lock:
            ids=tuple(request_ids)
            allocations=[self._get(i) for i in ids]
            width=max((len(a.blocks) for a in allocations),default=0)
            tables=torch.full((len(ids),width),-1,dtype=torch.int32)
            for row,a in enumerate(allocations):
                if a.blocks: tables[row,:len(a.blocks)]=torch.tensor(a.blocks,dtype=torch.int32)
            lengths=torch.tensor([a.length for a in allocations],dtype=torch.int32)
        target=self.keys.device if device is None else torch.device(device)
        return BatchKVMetadata(ids,tables.to(target),lengths.to(target),lengths.to(torch.int64).to(target))

    def write(self,layer,request_id,positions,key,value):
        """Correctness/reference writer; fused kernels should write slots directly."""
        allocation=self._get(request_id)
        if key.shape!=value.shape or key.shape!=(len(positions),self.num_kv_heads,self.head_dim):
            raise ValueError('K/V must be [tokens,kv_heads,head_dim]')
        if not 0<=layer<self.num_layers: raise IndexError('layer out of range')
        for source,position in enumerate(positions):
            if not 0<=position<allocation.length: raise IndexError('position not reserved')
            block=allocation.blocks[position//self.block_size]; offset=position&(self.block_size-1)
            self.keys[layer,block,:,offset].copy_(key[source])
            self.values[layer,block,:,offset].copy_(value[source])

    def length(self,request_id): return self._get(request_id).length
    def blocks(self,request_id): return tuple(self._get(request_id).blocks)
    @property
    def free_blocks(self): return len(self._free)
    @property
    def capacity_tokens(self): return self.num_blocks*self.block_size
    @property
    def storage_bytes(self): return self.keys.numel()*self.keys.element_size()+self.values.numel()*self.values.element_size()
    def blocks_for_tokens(self,count):
        if count<0: raise ValueError('token count must be nonnegative')
        return math.ceil(count/self.block_size)

    def _get(self,request_id):
        try: return self._requests[request_id]
        except KeyError: raise KeyError(f'unknown request {request_id!r}') from None
