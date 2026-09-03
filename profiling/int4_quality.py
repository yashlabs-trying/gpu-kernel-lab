"""Small teacher-forced decode quality check, not a standard corpus benchmark."""
import argparse
import json
import math
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from int4_model_ablation import install_int4_decode

TEXTS = [
    'A GPU executes many threads in parallel. Neighboring threads should read neighboring memory addresses. Shared memory lets cooperating threads reuse a tile, but synchronization is necessary before reading data written by another thread.',
    'Matrix multiplication computes each output as a sum of products. A tiled implementation reuses input values across several outputs. The tile size affects parallelism, register usage, and the amount of shared memory required.',
    'To add two vectors in Python, iterate over corresponding elements and add each pair. The result has the same length as the inputs. Check that both input vectors have matching lengths before computing the result.',
    'The train left the station early in the morning. Outside the window, the city slowly gave way to fields and small villages. By noon, the passengers could see mountains rising beyond the river.',
    'A rectangle has a width of seven meters and a length of nine meters. Its area is sixty-three square meters. Its perimeter is thirty-two meters because the sum of all four sides is seven plus nine plus seven plus nine.',
    'Water can exist as a solid, a liquid, or a gas. Heating ice causes it to melt. Further heating can turn liquid water into vapor. These changes depend on temperature and pressure.',
    'When testing an optimization, keep the workload and hardware fixed. Measure several repetitions after warming up the program. Compare both numerical output and execution time, and report cases where the proposed optimization is slower.',
    'An autoregressive language model predicts the next token from the preceding tokens. A key-value cache stores intermediate attention data from earlier positions. Reusing that cache avoids recomputing every previous token during generation.',
]


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B', local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B', dtype=torch.bfloat16,
        attn_implementation='sdpa', local_files_only=True).eval().cuda()
    tokens = [tok(text, return_tensors='pt')['input_ids'][:, :48].cuda() for text in TEXTS]

    def evaluate():
        rows = []
        for ids in tokens:
            cache = model(input_ids=ids[:, :8], use_cache=True, logits_to_keep=1).past_key_values
            values, choices = [], []
            for position in range(8, ids.shape[1]-1):
                out = model(input_ids=ids[:, position:position+1], past_key_values=cache,
                            use_cache=True, logits_to_keep=1)
                cache = out.past_key_values
                logits = out.logits[0, -1].float()
                values.append((-logits.log_softmax(0)[ids[0, position+1]]).item())
                choices.append(logits.argmax().item())
            rows.append({'nll':values, 'argmax':choices})
        return rows

    baseline = evaluate()
    install_int4_decode(model)
    quantized = evaluate()
    total = sum(len(row['nll']) for row in baseline)
    base_nll = sum(sum(row['nll']) for row in baseline) / total
    quant_nll = sum(sum(row['nll']) for row in quantized) / total
    agreement = sum(a == b for x,y in zip(baseline,quantized)
                    for a,b in zip(x['argmax'],y['argmax'])) / total
    report = {'method':'teacher-forced single-token decode on eight original short passages; not a standard corpus or acceptance threshold',
              'evaluated_tokens':total, 'baseline_mean_nll':base_nll,
              'int4_mean_nll':quant_nll, 'nll_delta':quant_nll-base_nll,
              'baseline_exp_nll':math.exp(base_nll), 'int4_exp_nll':math.exp(quant_nll),
              'teacher_forced_argmax_agreement':agreement,
              'texts':TEXTS, 'baseline':baseline, 'int4':quantized}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('texts','baseline','int4')},indent=2))


if __name__ == '__main__':
    main()
