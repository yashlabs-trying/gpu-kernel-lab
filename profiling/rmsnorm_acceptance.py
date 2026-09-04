"""RMSNorm numerical gate: adversarial tensors plus per-layer model drift."""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'learning/03_rmsnorm'))
from qwen_rmsnorm import qwen_rmsnorm_reference,triton_qwen_rmsnorm
sys.path.insert(0,str(ROOT/'learning/06_gemm'))
from prefill_dispatch import install_prefill_gemm_dispatch
sys.path.insert(0,str(ROOT/'learning/11_static_kv'))
from static_kv import advance_position,install_fused_static_kv,make_mask,new_static_cache
sys.path.insert(0,str(ROOT/'learning/12_decode_runtime'))
from decode_runtime import install_decode_residual_norm
from model_ablation import install


def metrics(actual,expected):
    delta=(actual.float()-expected.float()).abs()
    return {'max_abs':delta.max().item(),'mean_abs':delta.mean().item(),
        'cosine':F.cosine_similarity(actual.float().flatten(),expected.float().flatten(),dim=0).item()}


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(); p.add_argument('--prompts',type=int,default=256)
    p.add_argument('--output',type=Path,required=True); args=p.parse_args()
    torch.manual_seed(20260904)

    tensor_rows=[]
    for batch in (1,2,4):
        for context in (1,17,128,512):
            for scale in (1e-4,1e-2,1.0,100.0,1e4):
                x=torch.randn((batch,context,1024),device='cuda',dtype=torch.bfloat16)*scale
                w=torch.randn(1024,device='cuda',dtype=torch.bfloat16)
                expected=qwen_rmsnorm_reference(x,w); actual=triton_qwen_rmsnorm(x,w)
                tensor_rows.append({'batch':batch,'context':context,'scale':scale,**metrics(actual,expected)})

    model=AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B',dtype=torch.bfloat16,
        attn_implementation='sdpa',local_files_only=True).eval().cuda()
    vocab=model.config.vocab_size; context=16; capacity=context+2

    def collect(dynamic):
        layer_values=[]; handles=[]
        current=[]
        for index,layer in enumerate(model.model.layers):
            def save(_module,_inputs,output,index=index):
                if capture[0]: current.append((index,output.detach().clone()))
            handles.append(layer.register_forward_hook(save))
        rows=[]
        try:
            for prompt in range(args.prompts):
                ids=((torch.arange(context,device='cuda')*(prompt*2+1)+prompt*97+100)%vocab)[None].long()
                capture=[False]
                if dynamic:
                    out=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=True,logits_to_keep=1)
                    cache=out.past_key_values; token=ids[:,-1:]
                    capture[0]=True
                    out=model(input_ids=token,attention_mask=torch.ones((1,context+1),device='cuda',dtype=torch.long),
                        past_key_values=cache,use_cache=True,logits_to_keep=1)
                else:
                    cache=new_static_cache(model,capacity)
                    out=model(input_ids=ids,attention_mask=torch.ones_like(ids),past_key_values=cache,use_cache=True,logits_to_keep=1)
                    position=torch.tensor([context],device='cuda',dtype=torch.long)
                    mask=make_mask(1,capacity,context,torch.bfloat16,'cuda'); capture[0]=True
                    out=model(input_ids=ids[:,-1:],attention_mask={'full_attention':mask},position_ids=position.view(1,1),
                        past_key_values=cache,use_cache=True,logits_to_keep=1); advance_position(position)
                rows.append({'layers':[value for _,value in sorted(current)],'logits':out.logits.detach().clone()})
                current.clear()
        finally:
            for handle in handles: handle.remove()
        return rows

    baseline=collect(True)
    install_prefill_gemm_dispatch(model)
    install(model,'qwen_rms_cuda_hybrid')
    install_fused_static_kv(model,capacity,split_attention=True)
    install_decode_residual_norm(model)
    candidate=collect(False)

    layers=[]
    for index in range(len(model.model.layers)):
        pairs=[metrics(c['layers'][index],b['layers'][index]) for b,c in zip(baseline,candidate)]
        layers.append({'layer':index,'max_abs':max(x['max_abs'] for x in pairs),
            'mean_abs':sum(x['mean_abs'] for x in pairs)/len(pairs),
            'min_cosine':min(x['cosine'] for x in pairs)})
    logits=[metrics(c['logits'],b['logits']) for b,c in zip(baseline,candidate)]
    agreement=sum(c['logits'].argmax().item()==b['logits'].argmax().item() for b,c in zip(baseline,candidate))/len(baseline)
    report={'gpu':torch.cuda.get_device_name(),'prompts':args.prompts,
        'adversarial_cases':len(tensor_rows),'adversarial':tensor_rows,
        'adversarial_max_abs':max(x['max_abs'] for x in tensor_rows),
        'per_layer':layers,'logits':{'max_abs':max(x['max_abs'] for x in logits),
            'mean_abs':sum(x['mean_abs'] for x in logits)/len(logits),
            'min_cosine':min(x['cosine'] for x in logits),'argmax_agreement':agreement},
        'gate':{'pass':agreement>=.99 and min(x['min_cosine'] for x in layers)>=.999,
            'requirements':'argmax >= 99%, every layer min cosine >= 0.999'},
        'limitations':'synthetic deterministic prompts; task-quality corpus evaluation remains separate'}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('adversarial','per_layer')},indent=2))
    if not report['gate']['pass']: raise SystemExit(2)


if __name__=='__main__': main()
