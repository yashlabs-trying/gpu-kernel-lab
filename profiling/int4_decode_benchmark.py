"""Probe real Qwen decode activations with INT4 projections and compact LM top-k.

Keeps the original model for A/B comparison: buffer compression is NOT resident
model-memory reduction. Never treats quantization drift as kernel error.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import triton.testing
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'learning/09_int4'))
from quantization import quantize_int4, dequantize_int4
from kernels import int4_gemv, int4_lm_head_topk


def stats(a, b):
    a, b = a.float(), b.float()
    delta = (a-b).abs()
    return {'max_abs': delta.max().item(), 'mean_abs': delta.mean().item(),
            'relative_l2': ((a-b).norm() / b.norm().clamp_min(1e-12)).item()}


def bench(fn):
    fn(); torch.cuda.synchronize()
    return float(triton.testing.do_bench(fn, warmup=100, rep=300))


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'results/int4_decode.json')
    parser.add_argument('--group-size', type=int, choices=[32,64,128], default=128)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit('CUDA required')
    model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B',
        dtype=torch.bfloat16, attn_implementation='sdpa', local_files_only=True).eval().cuda()
    tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B', local_files_only=True)
    inputs = tok('Explain how a GPU executes matrix multiplication.', return_tensors='pt').to('cuda')
    prefix = model(**inputs, use_cache=True, logits_to_keep=1)
    token = prefix.logits[:, -1].argmax(-1, keepdim=True)
    names = ['model.layers.0.self_attn.q_proj', 'model.layers.0.mlp.gate_proj',
             'model.layers.0.mlp.down_proj', 'lm_head']
    activations, modules, hooks = {}, dict(model.named_modules()), []
    for name in names:
        def capture(module, values, name=name):
            activations[name] = values[0].reshape(-1, values[0].shape[-1])[-1].clone()
        hooks.append(modules[name].register_forward_pre_hook(capture))
    model(input_ids=token, past_key_values=prefix.past_key_values, use_cache=True,
          logits_to_keep=1)
    for hook in hooks:
        hook.remove()
    report = {'gpu':torch.cuda.get_device_name(), 'torch':torch.__version__,
        'group_size':args.group_size, 'method':'hot microbenchmarks on one real decode activation',
        'warning':'RTN quantization, not calibrated; original model retained; no end-to-end gain claimed',
        'projections':[]}
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for name in names:
            x, weight = activations[name], modules[name].weight
            packed = quantize_int4(weight, args.group_size)
            dense = F.linear(x, weight)
            oracle = (dequantize_int4(packed) @ x.float()).to(x.dtype)
            actual = int4_gemv(x, packed)
            correct = torch.allclose(actual, oracle, rtol=.02, atol=.002)
            row = {'name':name, 'shape':list(weight.shape),
                'bf16_bytes':weight.numel()*weight.element_size(), 'packed_bytes':packed.storage_bytes,
                'kernel_vs_dequantized':stats(actual,oracle), 'kernel_pass':correct,
                'quantization_vs_bf16':stats(oracle,dense),
                'bf16_ms':bench(lambda:F.linear(x,weight)),
                'int4_ms':bench(lambda:int4_gemv(x,packed)) if correct else None}
            if name == 'lm_head':
                row['top1_matches_bf16'] = actual.argmax().item() == dense.argmax().item()
                row['selection'] = []
                for top in (1,4,8):
                    values, ids = int4_lm_head_topk(x,packed,top)
                    matches = torch.equal(values, actual.float().topk(top).values)
                    row['selection'].append({'top_k':top,'matches_int4_full_logits':matches,
                        'ids':ids.tolist(), 'fused_ms':bench(lambda:int4_lm_head_topk(x,packed,top)) if matches and correct else None,
                        'bf16_head_plus_topk_ms':bench(lambda:F.linear(x,weight).float().topk(top))})
            report['projections'].append(row)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
