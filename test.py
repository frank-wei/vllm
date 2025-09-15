from vllm import LLM, SamplingParams

prompts = [
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(
    model="/data/local/model/gpt-oss-120b",
    tensor_parallel_size=4,
    speculative_config={
        "model": "/data/local/model/gpt-oss-120b-Eagle3-nvidia",
        "draft_tensor_parallel_size": 1,
        "num_speculative_tokens": 2,
        "method": "eagle",
    },
)

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
