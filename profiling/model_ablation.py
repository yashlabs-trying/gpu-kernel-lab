"""Process-local substitutions; record drift rather than claiming equivalence."""
import sys
import types
from pathlib import Path
import torch


def install(model, variant):
    root=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root/'learning/03_rmsnorm'))
    sys.path.insert(0,str(root/'learning/02_swiglu'))
    from rmsnorm import triton_rmsnorm
    from qwen_rmsnorm import triton_qwen_rmsnorm
    from swiglu import triton_swiglu
    from fused_projection_swiglu import fused_projection_swiglu
    count=0
    for module in model.modules():
        if variant in ('triton_rms','triton_both','qwen_rms') and type(module).__name__=='Qwen3RMSNorm':
            def forward(self, hidden_states):
                kernel = triton_qwen_rmsnorm if variant == 'qwen_rms' else triton_rmsnorm
                return kernel(hidden_states,self.weight,self.variance_epsilon)
            module.forward=types.MethodType(forward,module)
            count+=1
        if variant in ('triton_swiglu','triton_both','fused_mlp') and type(module).__name__=='Qwen3MLP':
            def forward(self, x):
                if variant == 'fused_mlp':
                    hidden=fused_projection_swiglu(x,self.gate_proj.weight,self.up_proj.weight)
                else:
                    hidden=triton_swiglu(self.gate_proj(x),self.up_proj(x))
                return self.down_proj(hidden)
            module.forward=types.MethodType(forward,module)
            count+=1
    return count


def validate_and_install(model,variant,lengths):
    def logits(length):
        ids=(torch.arange(length,device='cuda')[None,:]%10000+100).long()
        return model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=True,logits_to_keep=1).logits.float().clone()
    def greedy():
        ids=(torch.arange(128,device='cuda')[None,:]%10000+100).long()
        return model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),max_new_tokens=16,do_sample=False)[:,-16:].tolist()
    with torch.inference_mode():
        reference={length:logits(length) for length in lengths}
        reference_tokens=greedy()
        count=install(model,variant)
        rows=[]
        for length in lengths:
            actual=logits(length)
            expected=reference[length]
            rows.append({'length':length,'max_abs_logit_error':(actual-expected).abs().max().item(),
                         'mean_abs_logit_error':(actual-expected).abs().mean().item(),
                         'logit_cosine':torch.nn.functional.cosine_similarity(actual.flatten(),expected.flatten(),dim=0).item(),
                         'same_argmax':actual.argmax(-1).item()==expected.argmax(-1).item()})
        tokens=greedy()
    return {'substituted_modules':count,'last_token_logits':rows,
            'reference_greedy_16':reference_tokens,'candidate_greedy_16':tokens,
            'greedy_16_match':tokens==reference_tokens,
            'caution':'Small synthetic checks only. Legacy variants change rounding points; model-ordered candidates preserve explicit storage boundaries but GEMM/reduction order may differ. Not production-quality equivalence validation.'}
