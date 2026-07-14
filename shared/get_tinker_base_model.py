"""
WHAT THIS DOES.

TL;DR
1. Tinker is very reliable and good and pretty fast
2. Tinker doesn't let you do inference on BASE (i.e. non-finetuned) models
3. Here is a trick where we finetune-a-model w/o any finetuning so for tinker it's a finetuned model
   but we actually do inference on the non-finetuned model.

USAGE: set BASE_MODEL, run, your model name will be printed.
"""

import os

import tinker

BASE_MODEL = "openai/gpt-oss-120b"
LORA_RANK = 4
SUFFIX = "base-untrained"

# %%
if not os.environ.get("TINKER_API_KEY"):
    raise EnvironmentError(
        "TINKER_API_KEY environment variable is not set. "
        "Get an API key from https://tinker-console.thinkingmachines.ai/ "
        "and set it with: export TINKER_API_KEY='your-key'"
    )

service_client = tinker.ServiceClient()
training_client = service_client.create_lora_training_client(
    base_model=BASE_MODEL,
    rank=LORA_RANK,
)

name = f"{SUFFIX}-{BASE_MODEL.split('/')[-1]}"
result = training_client.save_weights_for_sampler(name=name).result()
model_path = result.path

print(f"Base model for inference: {model_path}")