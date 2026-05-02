from vllm import LLM, SamplingParams
import os


def initialize_vLLM(**kwargs):
    llm = LLM(**kwargs)
    return llm


def vLLM_run(model, prompt, sampling_params):
    print("Processing...")
    outputs = model.generate(prompt, sampling_params)

    return outputs[0].outputs[0].text, outputs[0].outputs[0].logprobs
