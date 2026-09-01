"""Generation-style baseline and isolated CUDA-profiler capture windows."""
import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['benchmark', 'capture'], default='benchmark')
    p.add_argument('--phase', choices=['prefill', 'decode'], default='prefill')
    p.add_argument('--lengths', nargs='+', type=int, default=[128, 512, 2048, 4096])
    p.add_argument('--steps', type=int, default=16)
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--warmup', type=int, default=2)
    p.add_argument('--backend', choices=['sdpa', 'eager'], default='sdpa')
    p.add_argument('--output', type=Path)
    p.add_argument('--annotate', action='store_true')
    p.add_argument('--variant', choices=['baseline','triton_rms','triton_swiglu','triton_both','qwen_rms'], default='baseline')
    args = p.parse_args()
    model = AutoModelForCausalLM.from_pretrained(
        'Qwen/Qwen3-0.6B', dtype=torch.bfloat16,
        attn_implementation=args.backend, local_files_only=True,
    ).eval().cuda()
    torch.manual_seed(0)
    hooks = []
    report = {'torch': torch.__version__, 'transformers': transformers.__version__,
              'backend': model.config._attn_implementation, 'dtype': 'bfloat16',
              'gpu': torch.cuda.get_device_name(), 'config': model.config.to_dict(),
              'workload': 'batch=1; cache enabled; last-token logits; decode includes argmax and mask growth',
              'warmup': args.warmup, 'repeats': args.repeats, 'measurements': []}
    report['variant'] = args.variant
    if args.variant != 'baseline':
        from model_ablation import validate_and_install
        report['ablation_validation'] = validate_and_install(model,args.variant,args.lengths)
        print(json.dumps(report['ablation_validation']), flush=True)

    def annotate():
        for name, module in model.named_modules():
            if name and (not list(module.children()) or name.startswith('model.layers.') and name.count('.') == 2):
                def before(m, a, name=name):
                    torch.cuda.nvtx.range_push(name)
                hooks.append(module.register_forward_pre_hook(before))
                def after(m, a, output):
                    torch.cuda.nvtx.range_pop()
                hooks.append(module.register_forward_hook(after))

    with torch.inference_mode():
        for length in args.lengths:
            ids = (torch.arange(length, device='cuda')[None, :] % 10000 + 100).long()
            mask = torch.ones_like(ids)
            def prefill():
                return model(input_ids=ids, attention_mask=mask, use_cache=True, logits_to_keep=1)

            def decode(cache, token, count):
                current_mask = mask
                for _ in range(count):
                    current_mask = torch.cat([current_mask, torch.ones((1, 1), device='cuda', dtype=torch.long)], dim=1)
                    result = model(input_ids=token, attention_mask=current_mask,
                                   past_key_values=cache, use_cache=True, logits_to_keep=1)
                    cache = result.past_key_values
                    token = result.logits[:, -1:].argmax(-1)

            phases = ['prefill', 'decode'] if args.mode == 'benchmark' else [args.phase]
            for phase in phases:
                for _ in range(args.warmup):
                    result = prefill()
                    if phase == 'decode':
                        decode(result.past_key_values, result.logits[:, -1:].argmax(-1), args.steps)
                    del result
                torch.cuda.synchronize()
                gpu_samples, wall_samples = [], []
                count = args.repeats if args.mode == 'benchmark' else 1
                for _ in range(count):
                    if phase == 'decode':
                        prepared = prefill()
                        cache, token = prepared.past_key_values, prepared.logits[:, -1:].argmax(-1)
                        del prepared
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    if args.mode == 'capture':
                        if args.annotate:
                            annotate()
                        torch.cuda.cudart().cudaProfilerStart()
                        torch.cuda.nvtx.range_push(f'{phase}_S{length}')
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    t0 = time.perf_counter()
                    start.record()
                    if phase == 'prefill':
                        result = prefill()
                    else:
                        decode(cache, token, args.steps)
                    end.record()
                    end.synchronize()
                    wall = (time.perf_counter() - t0) * 1000
                    divisor = 1 if phase == 'prefill' else args.steps
                    gpu_samples.append(start.elapsed_time(end) / divisor)
                    wall_samples.append(wall / divisor)
                    if args.mode == 'capture':
                        torch.cuda.nvtx.range_pop()
                        torch.cuda.cudart().cudaProfilerStop()
                        for hook in hooks:
                            hook.remove()
                        hooks.clear()
                    if phase == 'prefill':
                        del result
                    else:
                        del cache, token
                row = {'phase': phase, 'initial_context': length,
                       'decode_steps': args.steps if phase == 'decode' else None,
                       'gpu_median_ms': statistics.median(gpu_samples),
                       'wall_median_ms': statistics.median(wall_samples),
                       'gpu_samples_ms': gpu_samples, 'wall_samples_ms': wall_samples,
                       'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30}
                report['measurements'].append(row)
                print(json.dumps(row), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
