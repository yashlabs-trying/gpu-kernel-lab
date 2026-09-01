"""Small natural-language drift probe, not a model-quality benchmark."""
import json
import argparse
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from model_ablation import install


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=['triton_both', 'qwen_rms', 'fused_mlp'], default='triton_both')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    tokenizer=AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B',local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B',dtype=torch.bfloat16,
            attn_implementation='sdpa',local_files_only=True).eval().cuda()
    prompts=[
        'Explain in one sentence why GPUs are useful for matrix multiplication.',
        'What is 17 multiplied by 23? Give the answer briefly.',
        'Write a Python function that returns the sum of a list of numbers.',
        'Summarize in one sentence: A kernel loaded the same data repeatedly from global memory. Tiling allowed it to reuse the data in shared memory and improved performance.',
    ]
    def run(prompt):
        inputs=tokenizer.apply_chat_template([{'role':'user','content':prompt}],
            add_generation_prompt=True,enable_thinking=False,return_dict=True,return_tensors='pt').to('cuda')
        out=model.generate(**inputs,max_new_tokens=32,do_sample=False,use_cache=True)
        tokens=out[0,inputs['input_ids'].shape[1]:].tolist()
        return {'tokens':tokens,'text':tokenizer.decode(tokens,skip_special_tokens=True)}
    with torch.inference_mode():
        baseline=[run(prompt) for prompt in prompts]
        install(model,args.variant)
        candidate=[run(prompt) for prompt in prompts]
    rows=[{'prompt':prompt,'baseline':base,args.variant:cand,'exact_tokens_match':base['tokens']==cand['tokens']}
          for prompt,base,cand in zip(prompts,baseline,candidate)]
    report={'method':'4 prompts, greedy, at most 32 new tokens; not a comprehensive quality evaluation',
            'exact_match_count':sum(row['exact_tokens_match'] for row in rows),'results':rows}
    report['variant'] = args.variant
    path=args.output or Path(f'results/natural_language_drift_{args.variant}.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
