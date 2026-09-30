"""Manual vLLM INT8 runtime smoke check; not a pytest test module."""
import traceback

from vllm import LLM, SamplingParams

try:
    llm = LLM(
        model="/workspace/models/llama-3.2-3b-hf",
        quantization="int8_per_channel_weight_only",
        dtype="float16",
        enforce_eager=True,
    )
    params = SamplingParams(max_tokens=8, temperature=0.0)
    output = llm.generate(["The capital of France is"], params, use_tqdm=False)
    print("INT8 OK", output[0].outputs[0].text)
except Exception:
    with open("/workspace/int8_err.txt", "w") as error_file:
        traceback.print_exc(file=error_file)
    print("ERR")
