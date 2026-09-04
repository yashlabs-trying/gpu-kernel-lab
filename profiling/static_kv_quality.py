"""Small held-out dynamic/static/fused cache equivalence probe."""
import argparse
import json
import sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'learning/11_static_kv'))
from static_kv import advance_position,install_fused_static_kv,make_mask,new_static_cache,open_mask
from int4_quality import TEXTS
from model_ablation import install
sys.path.insert(0,str(ROOT/'learning/12_decode_runtime'))
from decode_runtime import install_decode_residual_norm


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); p.add_argument('--hybrid',action='store_true'); p.add_argument('--split-attention',action='store_true'); p.add_argument('--residual-norm',action='store_true'); p.add_argument('--protected-mlp-layers',type=int,nargs='*',default=[]); args=p.parse_args()
    model=AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B',dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).eval().cuda()
    tokenizer=AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B',local_files_only=True)
    if args.hybrid and args.protected_mlp_layers:
        install(model,'qwen_rms')
        from decode_stage2_ablation import install_stage2
        install_stage2(model,hybrid=True,protected_layers=args.protected_mlp_layers)
    else:
        install(model,'qwen_rms_cuda_hybrid' if args.hybrid else 'qwen_rms')
    token_sets=[tokenizer(text,return_tensors='pt').input_ids[:,:48].cuda() for text in TEXTS]
    capacity=max(ids.shape[1] for ids in token_sets)

    def evaluate(mode):
        losses,choices=[],[]
        for ids in token_sets:
            if mode=='dynamic':
                prefix_mask=torch.ones_like(ids[:,:8])
                out=model(input_ids=ids[:,:8],attention_mask=prefix_mask,use_cache=True,logits_to_keep=1)
                cache=out.past_key_values
            else:
                cache=new_static_cache(model,capacity)
                out=model(input_ids=ids[:,:8],attention_mask=torch.ones_like(ids[:,:8]),past_key_values=cache,use_cache=True,logits_to_keep=1)
                position=torch.tensor([8],device='cuda',dtype=torch.long)
                mask=make_mask(1,capacity,8,torch.bfloat16,'cuda')
                mapping={'full_attention':mask}
                pointers=[(layer.keys.data_ptr(),layer.values.data_ptr()) for layer in cache.layers]
                mask_pointer,position_pointer=mask.data_ptr(),position.data_ptr()
            for pos in range(8,ids.shape[1]-1):
                if mode=='dynamic':
                    prefix_mask=torch.cat((prefix_mask,torch.ones((1,1),device='cuda',dtype=torch.long)),1)
                    out=model(input_ids=ids[:,pos:pos+1],attention_mask=prefix_mask,past_key_values=cache,use_cache=True,logits_to_keep=1)
                else:
                    if mode=='static': open_mask(mask,position)
                    out=model(input_ids=ids[:,pos:pos+1],attention_mask=mapping,position_ids=position.view(1,1),past_key_values=cache,use_cache=True,logits_to_keep=1)
                    advance_position(position)
                cache=out.past_key_values
                if mode!='dynamic':
                    assert pointers==[(layer.keys.data_ptr(),layer.values.data_ptr()) for layer in cache.layers]
                    assert mask.data_ptr()==mask_pointer and position.data_ptr()==position_pointer
                logits=out.logits[0,-1].float()
                losses.append((-logits.log_softmax(0)[ids[0,pos+1]]).item()); choices.append(logits.argmax().item())
        return {'mean_nll':sum(losses)/len(losses),'argmax':choices,'tokens':len(losses)}

    dynamic=evaluate('dynamic'); static=evaluate('static')
    install_fused_static_kv(model,capacity,split_attention=args.split_attention)
    if args.residual_norm: install_decode_residual_norm(model)
    fused=evaluate('fused')
    report={'method':'eight original passages, teacher-forced single-token decode; not a standard quality benchmark','hybrid':args.hybrid,'split_attention':args.split_attention,'residual_norm':args.residual_norm,'protected_mlp_layers':args.protected_mlp_layers,
        'dynamic_mean_nll':dynamic['mean_nll'],'static_mean_nll':static['mean_nll'],'fused_mean_nll':fused['mean_nll'],
        'static_argmax_agreement':sum(a==b for a,b in zip(dynamic['argmax'],static['argmax']))/dynamic['tokens'],
        'fused_argmax_agreement':sum(a==b for a,b in zip(dynamic['argmax'],fused['argmax']))/dynamic['tokens'],
        'fused_vs_static_argmax_agreement':sum(a==b for a,b in zip(static['argmax'],fused['argmax']))/dynamic['tokens'],
        'evaluated_tokens':dynamic['tokens'],'capacity':capacity}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))


if __name__=='__main__': main()
