"""Opt-in W8A16 combined-projection adapter; preserves dense prefill fallback."""
import sys
import types
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'learning/10_cuda_decode'))
from activation_int8 import quantize_int8
from cuda_decode import project, extension
sys.path.insert(0,str(ROOT/'learning/02_swiglu'))
from fused_projection_swiglu import fused_projection_swiglu

# Separate from the eight held-out quality passages. These small calibration
# prompts are an experiment, not a representative production calibration set.
CALIBRATION = [
    'Describe the differences between a compiler and an interpreter, with an example of each.',
    'A shop sells twelve boxes with six pencils in each box. Explain how to find the total number of pencils.',
    'Write a short description of a forest after rainfall, including the sounds and colors.',
    'Explain what happens when a program reads a file from a disk and processes its contents.',
]


@torch.inference_mode()
def collect_calibration(model):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B',local_files_only=True)
    examples, hooks = {}, []
    for name,module in model.named_modules():
        if isinstance(module,torch.nn.Linear) and name != 'lm_head':
            def capture(m,values,name=name):
                # 8 rows/prompt, including first and final token positions.
                x = values[0].reshape(-1,values[0].shape[-1])
                positions = torch.linspace(0,x.shape[0]-1,min(8,x.shape[0]),device=x.device).long()
                examples.setdefault(name,[]).append(x[positions].detach().clone())
            hooks.append(module.register_forward_pre_hook(capture))
    try:
        for text in CALIBRATION:
            ids = tokenizer(text,return_tensors='pt').to(model.device)
            model(**ids,use_cache=False,logits_to_keep=1)
    finally:
        for hook in hooks:
            hook.remove()
    return {name:torch.cat(rows) for name,rows in examples.items()}


@torch.inference_mode()
def install_stage2(model, *, hybrid=False, protected_layers=()):
    if getattr(model,'_kernellab_int8_info',None) is not None:
        raise ValueError('INT8 adapter already installed')
    extension()  # compile outside inference timing
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        protected_layers=frozenset(int(i) for i in protected_layers)
        if any(i<0 or i>=len(model.model.layers) for i in protected_layers):
            raise ValueError('protected layer index out of range')
        samples = collect_calibration(model)
        state = {'decode':False}
        def detect(module,positional,keywords):
            ids = keywords.get('input_ids',positional[0] if positional else None)
            state['decode'] = ids is not None and tuple(ids.shape)==(1,1)
        model.register_forward_pre_hook(detect,with_kwargs=True)
        calibration, storage, count = {}, 0, 0

        def pack(name,weight,quant=True):
            nonlocal storage
            if not quant:
                storage += weight.numel()*weight.element_size()
                calibration[name] = {'dtype':'BF16','bytes':weight.numel()*weight.element_size()}
                return weight
            packed = quantize_int8(weight,samples[name])
            storage += packed.storage_bytes
            calibration[name] = {'alpha':packed.alpha,'local_mse':packed.calibration_mse,'bytes':packed.storage_bytes}
            return packed

        def linear(module,packed):
            original = module.forward
            def forward(self,x):
                if not state['decode'] or x.numel()!=x.shape[-1]:
                    return original(x)
                return project(x.reshape(-1),packed).reshape(*x.shape[:-1],packed.data.shape[0])
            module.forward = types.MethodType(forward,module)

        for index,layer in enumerate(model.model.layers):
            prefix = f'model.layers.{index}'
            attention, mlp = layer.self_attn, layer.mlp
            if any(m.bias is not None for m in (attention.q_proj,attention.k_proj,attention.v_proj,attention.o_proj,mlp.gate_proj,mlp.up_proj,mlp.down_proj)):
                raise ValueError('Qwen bias-free projections required')
            sizes = [m.out_features for m in (attention.q_proj,attention.k_proj,attention.v_proj)]
            qkv = pack(prefix+'.self_attn.q_proj',torch.cat([m.weight for m in (attention.q_proj,attention.k_proj,attention.v_proj)]),quant=not hybrid)
            # Closure-local cache reset at every attention call. q_proj executes
            # before k_proj and v_proj in the inspected Qwen3 implementation.
            def attach_qkv(attention,packed,sizes):
                pending = {}
                def reset(module,args):
                    pending.clear()
                attention.register_forward_pre_hook(reset)
                for slot,module in enumerate((attention.q_proj,attention.k_proj,attention.v_proj)):
                    original = module.forward
                    def forward(self,x,slot=slot,original=original):
                        if not state['decode'] or x.numel()!=x.shape[-1]:
                            return original(x)
                        if slot == 0:
                            outputs = project(x.reshape(-1),packed).split(sizes)
                            pending.update({i:y for i,y in enumerate(outputs)})
                        if slot not in pending:
                            raise RuntimeError('unexpected Qwen Q/K/V call order')
                        return pending.pop(slot).reshape(*x.shape[:-1],sizes[slot])
                    module.forward = types.MethodType(forward,module)
            attach_qkv(attention,qkv,sizes)
            if not hybrid:
                linear(attention.o_proj,pack(prefix+'.self_attn.o_proj',attention.o_proj.weight))
            protect=index in protected_layers
            gu = pack(prefix+'.mlp.gate_proj',torch.cat((mlp.gate_proj.weight,mlp.up_proj.weight)),quant=not protect)
            if not hybrid:
                linear(mlp.down_proj,pack(prefix+'.mlp.down_proj',mlp.down_proj.weight))
            def attach_mlp(mlp,packed):
                original = mlp.forward
                def forward(self,x):
                    rows=x.numel()//x.shape[-1]
                    if not state['decode']:
                        # Prefill has enough M parallelism for tensor-core tiles.
                        # Keep the decode-only quantized weights out of this path.
                        if rows > 4096:
                            return original(x)
                        if rows == 512:
                            # Let the exact-shape dispatcher use its measured
                            # gate/up winners at this one validated M value.
                            hidden=F.silu(self.gate_proj(x))*self.up_proj(x)
                            return self.down_proj(hidden)
                        hidden=fused_projection_swiglu(
                            x,self.gate_proj.weight,self.up_proj.weight)
                        return self.down_proj(hidden)
                    if rows != 1:
                        return original(x)
                    # Decode is M=1: stream packed gate/up weights once through
                    # independent warp reductions, then apply the SwiGLU epilogue.
                    hidden = project(x.reshape(-1),packed,swiglu=True,warps=8 if hybrid else 4).reshape(*x.shape[:-1],packed.shape[0]//2 if isinstance(packed,torch.Tensor) else packed.data.shape[0]//2)
                    return self.down_proj(hidden)
                mlp.forward = types.MethodType(forward,mlp)
            attach_mlp(mlp,gu)
            count += 5 if hybrid else 7
        model._kernellab_int8_info = {'linear_modules':count,'packed_bytes_added':storage,
            'method':'W8A16 AWQ-inspired channel scale search, alpha 0/.5/1; not full AWQ',
            'protected_bf16_mlp_layers':sorted(protected_layers),
            'hybrid':hybrid,'layout':'prefill: tiled BF16 gate/up/SwiGLU; decode: W8 dual GEMV/SwiGLU; '+('BF16 combined QKV + original o/down' if hybrid else 'W8 combined QKV + o + down'),
            'calibration':calibration,'calibration_texts':CALIBRATION,
            'warning':'Dense weights retained. Decode only, M=1, BF16, eager, no concurrent forwards. LM head stays BF16 unless separate head adapter installed. Split-KV attention is NOT integrated.'}
        return count
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
