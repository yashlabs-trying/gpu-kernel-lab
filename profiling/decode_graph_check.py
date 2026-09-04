"""Check CUDA Graph replay against the same eager fixed-cache decode path."""
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'learning/11_static_kv'))
from static_kv import advance_position,install_fused_static_kv,make_mask,new_static_cache
sys.path.insert(0,str(ROOT/'learning/12_decode_runtime'))
from decode_runtime import GreedyDecodeGraph,install_decode_residual_norm
from model_ablation import install


@torch.inference_mode()
def main():
    model=AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B',dtype=torch.bfloat16,
        attn_implementation='sdpa',local_files_only=True).eval().cuda()
    install(model,'qwen_rms_cuda_hybrid')
    capacity,context,steps=160,128,16
    install_fused_static_kv(model,capacity,split_attention=True)
    install_decode_residual_norm(model)

    def prepare():
        ids=(torch.arange(context,device='cuda')[None,:]%10000+100).long()
        cache=new_static_cache(model,capacity)
        out=model(input_ids=ids,attention_mask=torch.ones_like(ids),past_key_values=cache,
                  use_cache=True,logits_to_keep=1)
        position=torch.tensor([context],device='cuda',dtype=torch.long)
        mask=make_mask(1,capacity,context,torch.bfloat16,'cuda')
        return cache,out.logits[:,-1:].argmax(-1),position,{'full_attention':mask}

    eager_cache,eager_token,eager_pos,eager_mask=prepare()
    graph_cache,graph_token,graph_pos,graph_mask=prepare()
    graph=GreedyDecodeGraph(model,graph_token,graph_cache,graph_mask,graph_pos)

    eager_tokens=[]
    # Graph construction executes one warmup. Capture records the second step;
    # it does not replay it until the first explicit graph.replay().
    for _ in range(steps+1):
        out=model(input_ids=eager_token,attention_mask=eager_mask,
            position_ids=eager_pos.view(1,1),past_key_values=eager_cache,
            use_cache=True,logits_to_keep=1)
        eager_token.copy_(out.logits[:,-1:].argmax(-1)); advance_position(eager_pos)
        eager_tokens.append(eager_token.item())
    graph_tokens=[]
    for _ in range(steps):
        graph_tokens.append(graph.replay().item())
    torch.cuda.synchronize()
    result={
        'steps_checked':steps,
        'tokens_match':graph_tokens==eager_tokens[1:],
        'eager_tokens':eager_tokens[1:],
        'graph_tokens':graph_tokens,
        'position_match':graph_pos.item()==eager_pos.item(),
        'final_position':graph_pos.item(),
        'fixed_addresses':graph.addresses==graph._addresses(),
    }
    print(json.dumps(result,indent=2))
    if not all(result[k] for k in ('tokens_match','position_match','fixed_addresses')):
        raise SystemExit(1)


if __name__=='__main__': main()
