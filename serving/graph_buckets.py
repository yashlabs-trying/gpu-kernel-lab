"""Shape-keyed CUDA Graph registry; no silent padding beyond registered buckets."""
from dataclasses import dataclass
import torch


@dataclass
class _Captured:
    graph:torch.cuda.CUDAGraph
    inputs:tuple[torch.Tensor,...]
    output:object


class CUDAGraphBuckets:
    def __init__(self,batch_sizes=(1,2,4,8,16),sequence_buckets=(128,512,2048,4096)):
        if tuple(sorted(set(batch_sizes)))!=tuple(batch_sizes) or tuple(sorted(set(sequence_buckets)))!=tuple(sequence_buckets):
            raise ValueError('buckets must be unique and increasing')
        self.batch_sizes=tuple(batch_sizes); self.sequence_buckets=tuple(sequence_buckets); self._graphs={}

    def select(self,batch_size,sequence_length):
        batch=next((x for x in self.batch_sizes if x>=batch_size),None)
        sequence=next((x for x in self.sequence_buckets if x>=sequence_length),None)
        if batch is None or sequence is None: return None
        return batch,sequence

    @torch.inference_mode()
    def capture(self,key,fn,*static_inputs):
        if key in self._graphs: raise ValueError('graph key already captured')
        if not torch.cuda.is_available() or not all(x.is_cuda for x in static_inputs): raise ValueError('CUDA inputs required')
        fn(*static_inputs); torch.cuda.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph): output=fn(*static_inputs)
        self._graphs[key]=_Captured(graph,tuple(static_inputs),output)

    @torch.inference_mode()
    def replay(self,key,*inputs):
        captured=self._graphs[key]
        if len(inputs)!=len(captured.inputs): raise ValueError('input count changed')
        for destination,source in zip(captured.inputs,inputs):
            if source.shape!=destination.shape or source.dtype!=destination.dtype: raise ValueError('graph input signature changed')
            destination.copy_(source)
        captured.graph.replay(); return captured.output

