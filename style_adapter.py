import torch
import comfy.model_management
import copy
from .ipadapter.utils import encode_image_masked
from .utils import tensor_to_size
from .ipadapter.utils import WEIGHT_TYPES

class AdapterStyle:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "adapterstyle_loader": ("ADAPTERSTYLELOADER",),
                "clip_vision": ("CLIP_VISION",),
                "model": ("MODEL", ),
                "style": ("IMAGE",),
                "composition": ("IMAGE",),
                "weight_style": ("FLOAT", { "default": 1.0, "min": -1, "max": 5, "step": 0.05 }),
                "weight_composition": ("FLOAT", { "default": 1.0, "min": -1, "max": 5, "step": 0.05 }),
                "weight_type": (WEIGHT_TYPES, ),
                "combine_embeds": (["concat", "add", "subtract", "average", "norm average"], {"default": "average"}),
                "start_at": ("FLOAT", { "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "end_at": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "embeds_scaling": (['V only', 'K+V', 'K+V w/ C penalty', 'K+mean(V) w/ C penalty'], ),
            },
            "optional": {
                "image_negative": ("IMAGE",),
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("ADAPTERPATCH",)
    RETURN_NAMES = ("adapter_patch", )
    FUNCTION = "apply_adapter"
    CATEGORY = "Alkaid/Adapter"

    @staticmethod
    def combine_embedding(type, img_cond_embeds):
        if type != "concat" and img_cond_embeds.shape[0] > 1:
            if type == "add":
                img_cond_embeds = torch.sum(img_cond_embeds, dim=0).unsqueeze(0)
            elif type == "subtract":
                img_cond_embeds = img_cond_embeds[0] - torch.mean(img_cond_embeds[1:], dim=0)
                img_cond_embeds = img_cond_embeds.unsqueeze(0)
            elif type == "average":
                img_cond_embeds = torch.mean(img_cond_embeds, dim=0).unsqueeze(0)
            elif type == "norm average":
                img_cond_embeds = torch.mean(img_cond_embeds / torch.norm(img_cond_embeds, dim=0, keepdim=True), dim=0).unsqueeze(0)
        return img_cond_embeds




    def apply_adapter(self, adapterstyle_loader, clip_vision, model, style, composition, weight_style, weight_composition, weight_type,
                      combine_embeds='average', embeds_scaling='V only', start_at=0.0, end_at=0.9,
                      image_negative=None, mask=None,):
        weight = { 0: weight_style, 1: weight_style, 2: weight_style, 3: weight_composition, 4: weight_style, 5: weight_style, 6: weight_style, 7: weight_style, 8: weight_style, 9: weight_style, 10: weight_style }
        dtype = torch.float16 if comfy.model_management.should_use_fp16() else torch.bfloat16 if comfy.model_management.should_use_bf16() else torch.float32
        device = comfy.model_management.get_torch_device()

        ipadapter = copy.deepcopy(adapterstyle_loader['adapter']).to(device, dtype=dtype)
        img_cond_embeds = encode_image_masked(clip_vision, style)
        # if image_composition is not None:
        img_comp_cond_embeds = encode_image_masked(clip_vision, composition)

        img_cond_embeds = img_cond_embeds.penultimate_hidden_states
        image_negative = image_negative if image_negative is not None else torch.zeros([1, 224, 224, 3])
        img_uncond_embeds = encode_image_masked(clip_vision, image_negative).penultimate_hidden_states
        img_comp_cond_embeds = img_comp_cond_embeds.penultimate_hidden_states
        
        img_uncond_embeds = tensor_to_size(img_uncond_embeds, img_cond_embeds.shape[0])

        img_cond_embeds = img_cond_embeds.to(device, dtype=dtype)
        img_uncond_embeds = img_uncond_embeds.to(device, dtype=dtype)
        # if img_comp_cond_embeds is not None:
        img_comp_cond_embeds = img_comp_cond_embeds.to(device, dtype=dtype)

        img_cond_embeds = self.combine_embedding(type=combine_embeds, img_cond_embeds=img_cond_embeds)
        img_comp_cond_embeds = self.combine_embedding(type=combine_embeds, img_cond_embeds=img_comp_cond_embeds)

        if mask is not None:
            mask = mask.to(device, dtype=dtype)

        cond, uncond = ipadapter.get_image_embeds(img_cond_embeds, img_uncond_embeds)
        cond_comp = ipadapter.get_image_embeds(img_comp_cond_embeds, img_uncond_embeds)[0]

        cond = cond.to(device, dtype=dtype)
        uncond = uncond.to(device, dtype=dtype)
        cond_alt = None
        if img_comp_cond_embeds is not None:
            cond_alt = { 3: cond_comp.to(device, dtype=dtype) }

        sigma_start = model.model.model_sampling.percent_to_sigma(start_at)
        sigma_end = model.model.model_sampling.percent_to_sigma(end_at)

        patch_kwargs = {
            "ipadapter": ipadapter,
            "number": 0,
            "weight": weight,
            "cond": cond,
            "cond_alt": cond_alt,
            "uncond": uncond,
            "weight_type": weight_type,
            "mask": mask,
            "sigma_start": sigma_start,
            "sigma_end": sigma_end,
            "unfold_batch": False,
            "embeds_scaling": embeds_scaling,
        }

        return ({"style": patch_kwargs,}, )