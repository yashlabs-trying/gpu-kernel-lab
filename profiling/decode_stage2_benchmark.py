"""Real activation projection comparisons and synthetic split-KV attention."""
import argparse
import json
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import triton.testing
from transformers import AutoModelForCausalLM, AutoTokenizer
from decode_stage2_ablation import collect_calibration

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'learning/10_cuda_decode'))
from cuda_decode import project, extension
from activation_int8 import quantize_int8, dequantize_int8
from split_kv import split_kv_attention


def bench(fn):
    fn()
    return float(triton.testing.do_bench(fn,warmup=50,rep=150))


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    torch.manual_seed(10)
    torch.backends.cuda.matmul.allow_tf32 = False
    extension()
    model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B',dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).eval().cuda()
    calibration = collect_calibration(model)
    names = ['model.layers.0.self_attn.'+p+'_proj' for p in ('q','k','v','o')]+['model.layers.0.mlp.'+p+'_proj' for p in ('gate','up','down')]+['lm_head']
    modules, captured, hooks = dict(model.named_modules()), {}, []
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B',local_files_only=True)
    ids = tokenizer('Why does reusing matrix tiles reduce memory traffic?',return_tensors='pt').to('cuda')
    prefix = model(**ids,use_cache=True,logits_to_keep=1)
    for name in names:
        def capture(module,values,name=name):
            captured[name] = values[0].reshape(-1,values[0].shape[-1])[-1].clone()
        hooks.append(modules[name].register_forward_pre_hook(capture))
    model(input_ids=prefix.logits[:,-1].argmax(-1,keepdim=True),past_key_values=prefix.past_key_values,use_cache=True,logits_to_keep=1)
    for hook in hooks:
        hook.remove()
    report = {'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,
        'method':'hot M=1 microbenchmarks; calibration prompts separate from captured test activation; no integrated speed claim',
        'projections':[],'attention':[]}
    tasks = [(name,[name],False) for name in names if name!='lm_head']
    tasks += [('combined_qkv',names[:3],False),('fused_gate_up_swiglu',names[4:6],True)]
    for label, parts, swiglu in tasks:
        x = captured[parts[0]]
        weight = torch.cat([modules[n].weight for n in parts])
        packed = quantize_int8(weight,calibration[parts[0]])
        oracle = (dequantize_int8(packed)@x.float()).bfloat16()
        def reference():
            outputs = [F.linear(x,modules[n].weight) for n in parts]
            return F.silu(outputs[0])*outputs[1] if swiglu else (outputs[0] if len(parts)==1 else tuple(outputs))
        dense = torch.cat(reference()) if len(parts)>1 and not swiglu else reference()
        if swiglu:
            a,b = oracle.chunk(2)
            oracle = F.silu(a)*b
        row = {'name':label,'shape':list(weight.shape),'baseline_ms':bench(reference),
            'alpha':packed.alpha,'calibration_mse':packed.calibration_mse,
            'quantization_relative_l2':((oracle.float()-dense.float()).norm()/dense.float().norm().clamp_min(1e-12)).item(),
            'configs':[]}
        for quant in (False,True):
            target = packed if quant else weight
            expected = oracle if quant else (weight.float()@x.float()).bfloat16()
            if swiglu and not quant:
                a,b = expected.chunk(2)
                expected = F.silu(a)*b
            for warps in (1,4,8):
                for persistent in (False,True):
                    fn = lambda:project(x,target,warps=warps,persistent=persistent,swiglu=swiglu)
                    actual = fn()
                    correct = torch.allclose(actual,expected,atol=.02,rtol=.02)
                    row['configs'].append({'int8':quant,'warps':warps,'persistent':persistent,
                        'correct':correct,'max_abs_error':(actual.float()-expected.float()).abs().max().item(),
                        'ms':bench(fn) if correct else None})
        report['projections'].append(row)
        print(label,'baseline',row['baseline_ms'],'best',min((r for r in row['configs'] if r['correct']),key=lambda r:r['ms']),flush=True)
    del model,calibration,modules,prefix,captured
    for length in (128,2048,8192):
        q = torch.randn(16,128,device='cuda',dtype=torch.bfloat16)
        k = torch.randn(8,length,128,device='cuda',dtype=torch.bfloat16)
        v = torch.randn_like(k)
        # Single final-position query sees ALL supplied cache entries: causal=False.
        def sdpa():
            return F.scaled_dot_product_attention(q[None,:,None,:],k[None],v[None],enable_gqa=True,is_causal=False)[0,:,0]
        expected = sdpa()
        row = {'length':length,'sdpa_ms':bench(sdpa),'configs':[]}
        for tile in (64,128,256):
            fn = lambda:split_kv_attention(q,k,v,tile)
            actual = fn()
            correct = torch.allclose(actual,expected,atol=.003,rtol=.03)
            row['configs'].append({'tile':tile,'correct':correct,'max_abs_error':(actual-expected).abs().max().item(),'ms':bench(fn) if correct else None})
        report['attention'].append(row)
        print('attention',row,flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
