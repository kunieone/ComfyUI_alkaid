import comfy.model_management as model_management
from comfy.clip_vision import clip_preprocess, Output
from .CrossAttentionPatch import CrossAttentionPatch

WEIGHT_TYPES = ["linear", "ease in", "ease out", 'ease in-out', 'reverse in-out', 'weak input', 'weak output', 'weak middle', 'strong middle', 'style transfer (SDXL)', 'composition (SDXL)']


def set_model_patch_replace(model, patch_kwargs, key):
    to = model.model_options["transformer_options"]
    if "patches_replace" not in to:
        to["patches_replace"] = {}
    if "attn2" not in to["patches_replace"]:
        to["patches_replace"]["attn2"] = {}
    if key not in to["patches_replace"]["attn2"]:
        to["patches_replace"]["attn2"][key] = CrossAttentionPatch(**patch_kwargs)
    else:
        to["patches_replace"]["attn2"][key].set_new_condition(**patch_kwargs)

def encode_image_masked(clip_vision, image, mask=None):
    model_management.load_model_gpu(clip_vision.patcher)
    image = image.to(clip_vision.load_device)


    pixel_values = clip_preprocess(image.to(clip_vision.load_device)).float()

    if mask is not None:
        pixel_values = pixel_values * mask.to(clip_vision.load_device)

    out = clip_vision.model(pixel_values=pixel_values, intermediate_output=-2)

    outputs = Output()
    outputs["last_hidden_state"] = out[0].to(model_management.intermediate_device())
    outputs["image_embeds"] = out[2].to(model_management.intermediate_device())
    outputs["penultimate_hidden_states"] = out[1].to(model_management.intermediate_device())
    return outputs