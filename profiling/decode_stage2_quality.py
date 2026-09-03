"""Held-out teacher-forced decode and chat drift for selective candidates."""
import argparse
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from int4_quality import TEXTS
from model_ablation import install


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--variant',choices=['qwen_rms_int4_head','qwen_rms_int8_cuda','qwen_rms_int8_cuda_int4_head'],required=True)
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B',dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).eval().cuda()
    tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B',local_files_only=True)
    prompts = ['Explain in one sentence why GPUs are useful for matrix multiplication.',
        'What is 17 multiplied by 23? Give the answer briefly.',
        'Write a Python function that returns the sum of a list of numbers.',
        'Summarize in one sentence: A kernel loaded the same data repeatedly from global memory. Tiling allowed it to reuse the data in shared memory and improved performance.']
    def evaluate():
        losses, choices = [], []
        for text in TEXTS:
            ids = tok(text,return_tensors='pt').input_ids[:,:48].cuda()
            cache = model(input_ids=ids[:,:8],use_cache=True,logits_to_keep=1).past_key_values
            for pos in range(8,ids.shape[1]-1):
                out = model(input_ids=ids[:,pos:pos+1],past_key_values=cache,use_cache=True,logits_to_keep=1)
                cache = out.past_key_values
                logits = out.logits[0,-1].float()
                losses.append((-logits.log_softmax(0)[ids[0,pos+1]]).item())
                choices.append(logits.argmax().item())
        chats = []
        for prompt in prompts:
            inputs = tok.apply_chat_template([{'role':'user','content':prompt}],add_generation_prompt=True,enable_thinking=False,return_tensors='pt',return_dict=True).to('cuda')
            output = model.generate(**inputs,max_new_tokens=32,do_sample=False,use_cache=True)[0,inputs.input_ids.shape[1]:]
            chats.append({'tokens':output.tolist(),'text':tok.decode(output,skip_special_tokens=True)})
        return {'nll':losses,'argmax':choices,'chats':chats}
    baseline = evaluate()
    install(model,args.variant)
    candidate = evaluate()
    total = len(baseline['nll'])
    report = {'variant':args.variant,'method':'247 held-out teacher-forced tokens and four greedy chats; not a standard quality benchmark',
        'evaluated_tokens':total,'baseline_mean_nll':sum(baseline['nll'])/total,'candidate_mean_nll':sum(candidate['nll'])/total,
        'argmax_agreement':sum(a==b for a,b in zip(baseline['argmax'],candidate['argmax']))/total,
        'exact_chat_matches':sum(a['tokens']==b['tokens'] for a,b in zip(baseline['chats'],candidate['chats'])),
        'baseline':baseline,'candidate':candidate,'prompts':prompts,
        'int8_info':getattr(model,'_kernellab_int8_info',None)}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('baseline','candidate','int8_info','prompts')},indent=2))


if __name__=='__main__':
    main()
