import traceback
from vllm import LLM, SamplingParams
try:
    llm = LLM(model="/workspace/models/llama-3.2-3b-hf",
              quantization="int8_per_channel_weight_only",
              dtype="float16", enforce_eager=True)
    sp = SamplingParams(max_tokens=8, temperature=0.0)
    out = llm.generate(["The capital of France is"], sp, use_tqdm=False)
    print("INT8 OK", out[0].outputs[0].text)
except Exception:
    with open("/workspace/int8_err.txt", "w") as f:
        traceback.print_exc(file=f)
    print("ERR")
