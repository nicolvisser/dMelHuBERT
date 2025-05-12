import torch
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

from dmelhubert.dmelhubert import DMelHuBERT, DMelHuBERTArgs
from dmelhubert.dmelhubert_ctc import DMelHuBERTCTCArgs, DMelHuBERTCTCWithLora

model_args_without_lora = DMelHuBERTArgs.load_json(
    "/mnt/wsl/nvme/code/dMelHuBERT/checkpoints/dmelhubert-iter3/model_args.json"
)
model_without_lora = DMelHuBERT(model_args_without_lora)

checkpoint = torch.load(
    "/mnt/wsl/nvme/code/dMelHuBERT/checkpoints/dmelhubert-iter3/epoch=34-step=100000.ckpt",
    map_location="cpu",
)
state_dict = checkpoint["state_dict"]

del state_dict["model.proj.weight"]
del state_dict["model.proj.bias"]
del state_dict["model.label_embedding.weight"]
del state_dict["model.masked_spec_embed"]

model_without_lora.load_state_dict(state_dict, strict=False)
model_without_lora_keys = model_without_lora.state_dict().keys()
model_without_lora_num_keys = len(model_without_lora_keys)


model_with_lora = DMelHuBERTCTCWithLora(
    DMelHuBERTCTCArgs(),
    r=64,
    alpha=16.0,
    dropout=0.1,
    add_to_query=True,
    add_to_key=False,
    add_to_value=True,
    add_to_output=False,
)

model_with_lora_keys = model_with_lora.state_dict().keys()
model_with_lora_num_keys = len(model_with_lora_keys)

consume_prefix_in_state_dict_if_present(state_dict, "model.")
model_with_lora.load_state_dict(state_dict, strict=False)

trainable_keys = []
for key_name in model_with_lora.state_dict().keys():
    if "lora" in key_name:
        trainable_keys.append(key_name)
    if key_name == "proj.weight" or key_name == "proj.bias":
        trainable_keys.append(key_name)

print(len(trainable_keys))
