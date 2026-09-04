"""Deterministic, per-request logits processing and sampling."""
from dataclasses import dataclass
import math
import torch


@dataclass(frozen=True)
class SamplingParams:
    temperature:float=0.0
    top_k:int=0
    top_p:float=1.0
    repetition_penalty:float=1.0
    stop_token_ids:frozenset[int]=frozenset()
    seed:int=0
    logprobs:int=0

    def __post_init__(self):
        if self.temperature<0 or not math.isfinite(self.temperature): raise ValueError('temperature must be finite and nonnegative')
        if self.top_k<0 or not 0<self.top_p<=1: raise ValueError('invalid top-k/top-p')
        if self.repetition_penalty<=0 or not math.isfinite(self.repetition_penalty): raise ValueError('repetition penalty must be positive')
        if self.logprobs<0: raise ValueError('logprobs must be nonnegative')


@dataclass(frozen=True)
class SampleResult:
    token_id:int
    logprob:float
    top_logprobs:tuple[tuple[int,float],...]
    stopped:bool


class Sampler:
    def __init__(self,params:SamplingParams,device='cpu'):
        self.params=params; self.generator=torch.Generator(device=device); self.generator.manual_seed(params.seed)

    @torch.inference_mode()
    def sample(self,logits,history=()):
        if logits.ndim!=1 or not logits.is_floating_point(): raise ValueError('logits must be a floating vector')
        scores=logits.float().clone(); p=self.params
        if p.repetition_penalty!=1 and history:
            ids=torch.tensor(sorted(set(history)),device=scores.device,dtype=torch.long)
            selected=scores[ids]; scores[ids]=torch.where(selected<0,selected*p.repetition_penalty,selected/p.repetition_penalty)
        if p.temperature==0:
            token=int(scores.argmax())
        else:
            scores/=p.temperature
            if p.top_k:
                k=min(p.top_k,scores.numel()); cutoff=scores.topk(k).values[-1]; scores.masked_fill_(scores<cutoff,-torch.inf)
            if p.top_p<1:
                ordered,index=scores.sort(descending=True); probabilities=ordered.softmax(0)
                remove=probabilities.cumsum(0)-probabilities>p.top_p
                scores[index[remove]]=-torch.inf
            token=int(torch.multinomial(scores.softmax(0),1,generator=self.generator))
        log_probabilities=scores.log_softmax(0); count=min(p.logprobs,scores.numel())
        top=() if not count else tuple((int(i),float(v)) for v,i in zip(*log_probabilities.topk(count)))
        return SampleResult(token,float(log_probabilities[token]),top,token in p.stop_token_ids)

