import math
import copy

import torch
import torch.nn as nn

import comfy.model_management

from .utils import tensor_to_image, image_to_tensor
from .ipadapter.utils import WEIGHT_TYPES


class AdapterFace:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "adapterface_loader": ("ADAPTERFACELOADER",),
                "alkaid_loader": ("ALKAIDLOADER",),
                "model": ("MODEL", ),
                "image": ("IMAGE", ),
                "weight": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 5.0, "step": 0.01, }),
                "weight_type": (WEIGHT_TYPES, ),
                "ip_weight": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 3.0, "step": 0.01, }),
                "noise": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1, }),
                "combine_embeds": (['average', 'norm average', 'concat'], {"default": 'average'}),
                "embeds_scaling_id": (['V only', 'K+V', 'K+V w/ C penalty', 'K+mean(V) w/ C penalty'], ),
                "embeds_scaling_ip": (['V only', 'K+V', 'K+V w/ C penalty', 'K+mean(V) w/ C penalty'], ),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001, }),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001, }),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("ADAPTERPATCH", "ADAPTEREMB",)
    RETURN_NAMES = ("adapter_patch", "adapter_emb", )
    FUNCTION = "apply_adapter"
    CATEGORY = "Alkaid/Adapter"

    @staticmethod
    def extract_face_emb(insightface, image, is_norm=False):
        face_img = tensor_to_image(image)
        embedding = []
        insightface.det_model.input_size = (640,640) # reset the detection size

        for i in range(face_img.shape[0]):
            for size in [(size, size) for size in range(640, 128, -64)]:
                insightface.det_model.input_size = size # TODO: hacky but seems to be working
                face = insightface.get(face_img[i])
                if face:
                    face = sorted(face, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]
                    if is_norm:
                        embedding.append(torch.from_numpy(face.normed_embedding).unsqueeze(0))
                    else:
                        embedding.append(torch.from_numpy(face['embedding']).unsqueeze(0))
                    if 640 not in size:
                        print(f"\033[33mINFO: InsightFace detection resolution lowered to {size}.\033[0m")
                    break

        if embedding.__len__() != 0:
            embedding = torch.stack(embedding, dim=0)
        else:
            embedding = None

        return embedding

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


    def apply_adapter(self, adapterface_loader, alkaid_loader, model, image, weight, weight_type, ip_weight=None, 
                      noise=0.35, combine_embeds='average', embeds_scaling_id='V only', embeds_scaling_ip='V only', 
                      start_at=0.0, end_at=0.9,
                      mask=None,):
        

        dtype = torch.float16 if comfy.model_management.should_use_fp16() else torch.bfloat16 if comfy.model_management.should_use_bf16() else torch.float32
        device = comfy.model_management.get_torch_device()

        if mask is not None:
            mask = mask.to(device)

        '''------------------------'''
        face_cond_embeds = self.extract_face_emb(alkaid_loader['insightface_b'], image, is_norm=True)

        if face_cond_embeds is None:
            raise Exception('Reference Image: No face detected.')

        img_cond_embeds = face_cond_embeds       
        img_uncond_embeds = torch.zeros_like(img_cond_embeds)
        
        img_cond_embeds = img_cond_embeds.to(device, dtype=dtype)
        img_uncond_embeds = img_uncond_embeds.to(device, dtype=dtype)
        img_cond_embeds = self.combine_embedding(type=combine_embeds, img_cond_embeds=img_cond_embeds)
        img_uncond_embeds = img_uncond_embeds[0].unsqueeze(0) # TODO: better strategy for uncond could be to average them

        ipadapter = copy.deepcopy(adapterface_loader['ipadapter']).to(device, dtype=dtype)
        cond, uncond = ipadapter.get_image_embeds(img_cond_embeds, img_uncond_embeds)

        cond = cond.to(device, dtype=dtype)
        uncond = uncond.to(device, dtype=dtype)

        patch_kwargs_ip = {
            "ipadapter": ipadapter,
            "number": 0,
            "weight": weight,
            "cond": cond,
            "uncond": uncond,
            "weight_type": weight_type,
            "mask": mask,
            "sigma_start": model.model.model_sampling.percent_to_sigma(start_at),
            "sigma_end": model.model.model_sampling.percent_to_sigma(end_at),
            "embeds_scaling": embeds_scaling_ip,
        }

        '''------------------------'''

        face_embed = self.extract_face_emb(alkaid_loader['insightface'], image)
        if face_embed is None:
            raise Exception('Reference Image: No face detected.')

        clip_embed = face_embed
        # InstantID works better with averaged embeds (TODO: needs testing)

        clip_embed = self.combine_embedding(type=combine_embeds, img_cond_embeds=clip_embed)

        if noise > 0:
            seed = int(torch.sum(clip_embed).item()) % 1000000007
            torch.manual_seed(seed)
            clip_embed_zeroed = noise * torch.rand_like(clip_embed)
        else:
            clip_embed_zeroed = torch.zeros_like(clip_embed)

        instantid = copy.deepcopy(adapterface_loader['instantid']).to(device, dtype=dtype)
        image_prompt_embeds, uncond_image_prompt_embeds = instantid.get_image_embeds(clip_embed.to(device, dtype=dtype), clip_embed_zeroed.to(device, dtype=dtype))
        image_prompt_embeds = image_prompt_embeds.to(device, dtype=dtype)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.to(device, dtype=dtype)

        patch_kwargs_id = {
            "ipadapter": instantid,
            "number": 0,
            "weight": ip_weight,
            "cond": image_prompt_embeds,
            "uncond": uncond_image_prompt_embeds,
            "mask": mask,
            "sigma_start": model.model.model_sampling.percent_to_sigma(0),
            "sigma_end": model.model.model_sampling.percent_to_sigma(1),
            "embeds_scaling": embeds_scaling_id,
        }

        '''------------------------'''

        patch_kwargs = {
            'ip': patch_kwargs_ip,
            'id': patch_kwargs_id
        }
        embeddings = {
            'cond': image_prompt_embeds,
            'uncond': uncond_image_prompt_embeds
        }

        return(patch_kwargs, embeddings, )


class ApplyControlNet_KPS:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_kps": ("IMAGE",),
                "embedding": ("ADAPTEREMB",),
                "control_net": ("CONTROL_NET", ),
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 10.0, "step": 0.01, }),
                },
            "optional": {
                "mask": ("MASK",),
            }
        }
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING",)
    RETURN_NAMES = ("positive", "negative", )
    FUNCTION = "apply_controlnet"
    CATEGORY = "Alkaid/Adapter"

    def apply_controlnet(self, image_kps, embedding, control_net, positive, negative, strength, mask=None):

        face_kps = image_kps.to(comfy.model_management.intermediate_device())
        if mask is not None and len(mask.shape) < 3:
            mask = mask.unsqueeze(0)
        face_kps = face_kps[0:1]
        cnets = {}
        cond_uncond = []
        image_prompt_embeds = embedding['cond']
        uncond_image_prompt_embeds = embedding['uncond']
        is_cond = True
        for conditioning in [positive, negative]:
            c = []
            for t in conditioning:
                d = t[1].copy()

                prev_cnet = d.get('control', None)
                if prev_cnet in cnets:
                    c_net = cnets[prev_cnet]
                else:
                    c_net = control_net.copy().set_cond_hint(face_kps.movedim(-1,1), strength, (0, 1))
                    c_net.set_previous_controlnet(prev_cnet)
                    cnets[prev_cnet] = c_net

                d['control'] = c_net
                d['control_apply_to_uncond'] = False
                d['cross_attn_controlnet'] = image_prompt_embeds.to(comfy.model_management.intermediate_device()) if is_cond else uncond_image_prompt_embeds.to(comfy.model_management.intermediate_device())

                if mask is not None and is_cond:
                    d['mask'] = mask
                    d['set_area_to_bounds'] = False

                n = [t[0], d]
                c.append(n)
            cond_uncond.append(c)
            is_cond = False
        return (cond_uncond[0], cond_uncond[1], )
    

# WEIGHT_TYPES = ["linear", "ease in", "ease out", 'ease in-out', 'reverse in-out', 'weak input', 'weak output', 'weak middle', 'strong middle', 'style transfer (SDXL)', 'composition (SDXL)']

# # FFN
# def FeedForward(dim, mult=4):
#     inner_dim = int(dim * mult)
#     return nn.Sequential(
#         nn.LayerNorm(dim),
#         nn.Linear(dim, inner_dim, bias=False),
#         nn.GELU(),
#         nn.Linear(inner_dim, dim, bias=False),
#     )
    
    
# def reshape_tensor(x, heads):
#     bs, length, width = x.shape
#     #(bs, length, width) --> (bs, length, n_heads, dim_per_head)
#     x = x.view(bs, length, heads, -1)
#     # (bs, length, n_heads, dim_per_head) --> (bs, n_heads, length, dim_per_head)
#     x = x.transpose(1, 2)
#     # (bs, n_heads, length, dim_per_head) --> (bs*n_heads, length, dim_per_head)
#     x = x.reshape(bs, heads, length, -1)
#     return x


# class PerceiverAttention(nn.Module):
#     def __init__(self, *, dim, dim_head=64, heads=8):
#         super().__init__()
#         self.scale = dim_head**-0.5
#         self.dim_head = dim_head
#         self.heads = heads
#         inner_dim = dim_head * heads

#         self.norm1 = nn.LayerNorm(dim)
#         self.norm2 = nn.LayerNorm(dim)

#         self.to_q = nn.Linear(dim, inner_dim, bias=False)
#         self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
#         self.to_out = nn.Linear(inner_dim, dim, bias=False)


#     def forward(self, x, latents):
#         """
#         Args:
#             x (torch.Tensor): image features
#                 shape (b, n1, D)
#             latent (torch.Tensor): latent features
#                 shape (b, n2, D)
#         """
#         x = self.norm1(x)
#         latents = self.norm2(latents)
        
#         b, l, _ = latents.shape

#         q = self.to_q(latents)
#         kv_input = torch.cat((x, latents), dim=-2)
#         k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        
#         q = reshape_tensor(q, self.heads)
#         k = reshape_tensor(k, self.heads)
#         v = reshape_tensor(v, self.heads)

#         # attention
#         scale = 1 / math.sqrt(math.sqrt(self.dim_head))
#         weight = (q * scale) @ (k * scale).transpose(-2, -1) # More stable with f16 than dividing afterwards
#         weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
#         out = weight @ v
        
#         out = out.permute(0, 2, 1, 3).reshape(b, l, -1)

#         return self.to_out(out)


# class Resampler(nn.Module):
#     def __init__(
#         self,
#         dim=1024,
#         depth=8,
#         dim_head=64,
#         heads=16,
#         num_queries=8,
#         embedding_dim=768,
#         output_dim=1024,
#         ff_mult=4,
#     ):
#         super().__init__()
        
#         self.latents = nn.Parameter(torch.randn(1, num_queries, dim) / dim**0.5)
        
#         self.proj_in = nn.Linear(embedding_dim, dim)

#         self.proj_out = nn.Linear(dim, output_dim)
#         self.norm_out = nn.LayerNorm(output_dim)
        
#         self.layers = nn.ModuleList([])
#         for _ in range(depth):
#             self.layers.append(
#                 nn.ModuleList(
#                     [
#                         PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
#                         FeedForward(dim=dim, mult=ff_mult),
#                     ]
#                 )
#             )

#     def forward(self, x):
        
#         latents = self.latents.repeat(x.size(0), 1, 1)
        
#         x = self.proj_in(x)
        
#         for attn, ff in self.layers:
#             latents = attn(x, latents) + latents
#             latents = ff(latents) + latents
            
#         latents = self.proj_out(latents)
#         return self.norm_out(latents)

# class InstantID(torch.nn.Module):
#     def __init__(self, instantid_model, cross_attention_dim=1280, output_cross_attention_dim=1024, clip_embeddings_dim=512, clip_extra_context_tokens=16):
#         super().__init__()

#         self.clip_embeddings_dim = clip_embeddings_dim
#         self.cross_attention_dim = cross_attention_dim
#         self.output_cross_attention_dim = output_cross_attention_dim
#         self.clip_extra_context_tokens = clip_extra_context_tokens

#         self.image_proj_model = self.init_proj()

#         self.image_proj_model.load_state_dict(instantid_model["image_proj"])
#         self.ip_layers = InstantID_To_KV(instantid_model["ip_adapter"])

#     def init_proj(self):
#         image_proj_model = Resampler(
#             dim=self.cross_attention_dim,
#             depth=4,
#             dim_head=64,
#             heads=20,
#             num_queries=self.clip_extra_context_tokens,
#             embedding_dim=self.clip_embeddings_dim,
#             output_dim=self.output_cross_attention_dim,
#             ff_mult=4
#         )
#         return image_proj_model

#     @torch.inference_mode()
#     def get_image_embeds(self, clip_embed, clip_embed_zeroed):
#         #image_prompt_embeds = clip_embed.clone().detach()
#         image_prompt_embeds = self.image_proj_model(clip_embed)
#         #uncond_image_prompt_embeds = clip_embed_zeroed.clone().detach()
#         uncond_image_prompt_embeds = self.image_proj_model(clip_embed_zeroed)

#         return image_prompt_embeds, uncond_image_prompt_embeds

# class IDAdapterTo_KV(nn.Module):
#     def __init__(self, state_dict):
#         super().__init__()

#         self.to_kvs = nn.ModuleDict()
#         for key, value in state_dict.items():
#             self.to_kvs[key.replace(".weight", "").replace(".", "_")] = nn.Linear(value.shape[1], value.shape[0], bias=False)
#             self.to_kvs[key.replace(".weight", "").replace(".", "_")].weight.data = value


# class MLPProjModel(nn.Module):
#     def __init__(self, cross_attention_dim=768, id_embeddings_dim=512, num_tokens=4):
#         super().__init__()

#         self.cross_attention_dim = cross_attention_dim
#         self.num_tokens = num_tokens

#         self.proj = nn.Sequential(
#             nn.Linear(id_embeddings_dim, id_embeddings_dim*2),
#             nn.GELU(),
#             nn.Linear(id_embeddings_dim*2, cross_attention_dim*num_tokens),
#         )
#         self.norm = nn.LayerNorm(cross_attention_dim)

#     def forward(self, id_embeds):
#         x = self.proj(id_embeds)
#         x = x.reshape(-1, self.num_tokens, self.cross_attention_dim)
#         x = self.norm(x)
#         return x

# class IDAdapter(nn.Module):
#     def __init__(self, ipadapter_model, cross_attention_dim=1024, output_cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4,):
#         super().__init__()

#         self.clip_embeddings_dim = clip_embeddings_dim
#         self.cross_attention_dim = cross_attention_dim
#         self.output_cross_attention_dim = output_cross_attention_dim
#         self.clip_extra_context_tokens = clip_extra_context_tokens

#         self.image_proj_model = self.init_proj_faceid()
        

#         self.image_proj_model.load_state_dict(ipadapter_model["image_proj"])
#         self.ip_layers = IDAdapterTo_KV(ipadapter_model["ip_adapter"])


#     def init_proj_faceid(self):

#         image_proj_model = MLPProjModel(
#             cross_attention_dim=self.cross_attention_dim,
#             id_embeddings_dim=512,
#             num_tokens=self.clip_extra_context_tokens,
#         )
#         return image_proj_model

#     @torch.inference_mode()
#     def get_image_embeds(self, clip_embed, clip_embed_zeroed):
#         image_prompt_embeds = self.image_proj_model(clip_embed)
#         uncond_image_prompt_embeds = self.image_proj_model(clip_embed_zeroed)
#         return image_prompt_embeds, uncond_image_prompt_embeds

# class ImageProjModel(torch.nn.Module):
#     def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
#         super().__init__()

#         self.cross_attention_dim = cross_attention_dim
#         self.clip_extra_context_tokens = clip_extra_context_tokens
#         self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
#         self.norm = torch.nn.LayerNorm(cross_attention_dim)

#     def forward(self, image_embeds):
#         embeds = image_embeds
#         clip_extra_context_tokens = self.proj(embeds).reshape(-1, self.clip_extra_context_tokens, self.cross_attention_dim)
#         clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
#         return clip_extra_context_tokens

# class InstantID_To_KV(torch.nn.Module):
#     def __init__(self, state_dict):
#         super().__init__()

#         self.to_kvs = torch.nn.ModuleDict()
#         for key, value in state_dict.items():
#             k = key.replace(".weight", "").replace(".", "_")
#             self.to_kvs[k] = torch.nn.Linear(value.shape[1], value.shape[0], bias=False)
#             self.to_kvs[k].weight.data = value

# def _set_model_patch_replace(model, patch_kwargs, key):
#     to = model.model_options["transformer_options"]
#     if "patches_replace" not in to:
#         to["patches_replace"] = {}
#     if "attn2" not in to["patches_replace"]:
#         to["patches_replace"]["attn2"] = {}
#     if key not in to["patches_replace"]["attn2"]:
#         to["patches_replace"]["attn2"][key] = CrossAttentionPatch(**patch_kwargs)
#     else:
#         to["patches_replace"]["attn2"][key].set_new_condition(**patch_kwargs)



# class AdapterFace:
#     @classmethod
#     def INPUT_TYPES(s):
#         return {
#             "required": {
#                 "adapterface_loader": ("ADAPTERFACELOADER",),
#                 "alkaid_loader": ("ALKAIDLOADER",),
#                 "control_net": ("CONTROL_NET", ),
#                 "model": ("MODEL", ),
#                 "image": ("IMAGE", ),
#                 "image_kps": ("IMAGE",),
#                 "positive": ("CONDITIONING", ),
#                 "negative": ("CONDITIONING", ),
#                 "weight": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 5.0, "step": 0.01, }),
#                 "weight_type": (WEIGHT_TYPES, ),
#                 "ip_weight": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 3.0, "step": 0.01, }),
#                 "cn_strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 10.0, "step": 0.01, }),
#                 "noise": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1, }),
#                 "combine_embeds": (['average', 'norm average', 'concat'], {"default": 'average'}),
#                 "embeds_scaling_id": (['V only', 'K+V', 'K+V w/ C penalty', 'K+mean(V) w/ C penalty'], ),
#                 "embeds_scaling_ip": (['V only', 'K+V', 'K+V w/ C penalty', 'K+mean(V) w/ C penalty'], ),
#                 "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001, }),
#                 "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001, }),
#             },
#             "optional": {
#                 "mask": ("MASK",),
#             }
#         }

#     RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING",)
#     RETURN_NAMES = ("model", "positive", "negative", )
#     FUNCTION = "apply_adapter"
#     CATEGORY = "Alkaid/Tools"

#     @staticmethod
#     def extract_face_emb(insightface, image, is_norm=False):
#         face_img = tensor_to_image(image)
#         embedding = []
#         insightface.det_model.input_size = (640,640) # reset the detection size

#         for i in range(face_img.shape[0]):
#             for size in [(size, size) for size in range(640, 128, -64)]:
#                 insightface.det_model.input_size = size # TODO: hacky but seems to be working
#                 face = insightface.get(face_img[i])
#                 if face:
#                     face = sorted(face, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]
#                     if is_norm:
#                         embedding.append(torch.from_numpy(face.normed_embedding).unsqueeze(0))
#                     else:
#                         embedding.append(torch.from_numpy(face['embedding']).unsqueeze(0))
#                     if 640 not in size:
#                         print(f"\033[33mINFO: InsightFace detection resolution lowered to {size}.\033[0m")
#                     break

#         if embedding.__len__() != 0:
#             embedding = torch.stack(embedding, dim=0)
#         else:
#             embedding = None

#         return embedding

#     @staticmethod
#     def combine_embedding(type, img_cond_embeds):
#         if type != "concat" and img_cond_embeds.shape[0] > 1:
#             if type == "add":
#                 img_cond_embeds = torch.sum(img_cond_embeds, dim=0).unsqueeze(0)
#             elif type == "subtract":
#                 img_cond_embeds = img_cond_embeds[0] - torch.mean(img_cond_embeds[1:], dim=0)
#                 img_cond_embeds = img_cond_embeds.unsqueeze(0)
#             elif type == "average":
#                 img_cond_embeds = torch.mean(img_cond_embeds, dim=0).unsqueeze(0)
#             elif type == "norm average":
#                 img_cond_embeds = torch.mean(img_cond_embeds / torch.norm(img_cond_embeds, dim=0, keepdim=True), dim=0).unsqueeze(0)
#         return img_cond_embeds

#     def apply_adapter(self, adapterface_loader, alkaid_loader, control_net, model, image, image_kps, positive, negative, weight, weight_type, ip_weight=None, cn_strength=None,
#                       noise=0.35, combine_embeds='average', embeds_scaling_id='V only', embeds_scaling_ip='V only', 
#                       start_at=0.0, end_at=0.9,
#                       mask=None,):
        

#         print('''------------------------''')
#         dtype = torch.float16 if comfy.model_management.should_use_fp16() else torch.bfloat16 if comfy.model_management.should_use_bf16() else torch.float32
#         # dtype = torch.float16 if comfy.model_management.should_use_fp16() else torch.float32

#         device = comfy.model_management.get_torch_device()

#         if mask is not None:
#             mask = mask.to(device)

#         output_cross_attention_dim = 2048

#         work_model = model.clone()

#         work_model = work_model.clone()

#         face_cond_embeds = self.extract_face_emb(alkaid_loader['insightface_b'], image, is_norm=True)

#         if face_cond_embeds is None:
#             raise Exception('Reference Image: No face detected.')


#         img_cond_embeds = face_cond_embeds
       
#         img_uncond_embeds = torch.zeros_like(img_cond_embeds)
        
        
#         img_cond_embeds = img_cond_embeds.to(device, dtype=dtype)
#         img_uncond_embeds = img_uncond_embeds.to(device, dtype=dtype)
#         img_cond_embeds = self.combine_embedding(type=combine_embeds, img_cond_embeds=img_cond_embeds)
#         img_uncond_embeds = img_uncond_embeds[0].unsqueeze(0) # TODO: better strategy for uncond could be to average them

#         idadapter = IDAdapter(
#             adapterface_loader['ipadapter'],
#             cross_attention_dim=2048,
#             output_cross_attention_dim=output_cross_attention_dim,
#             clip_embeddings_dim=img_cond_embeds.shape[-1],
#             clip_extra_context_tokens=16,
#         ).to(device, dtype=dtype)

#         print(f'img_cond_embeds.shape[-1]: {img_cond_embeds.shape[-1]}')

#         cond, uncond = idadapter.get_image_embeds(img_cond_embeds, img_uncond_embeds)

#         cond = cond.to(device, dtype=dtype)
#         uncond = uncond.to(device, dtype=dtype)

#         patch_kwargs_ip = {
#             "ipadapter": idadapter,
#             "number": 0,
#             "weight": weight,
#             "cond": cond,
#             "uncond": uncond,
#             "weight_type": weight_type,
#             "mask": mask,
#             "sigma_start": work_model.model.model_sampling.percent_to_sigma(start_at),
#             "sigma_end": work_model.model.model_sampling.percent_to_sigma(end_at),
#             "embeds_scaling": embeds_scaling_ip,
#         }

#         '''------------------------'''

#         for id in [4,5,7,8]: # id of input_blocks that have cross attention
#             block_indices = range(2) if id in [4, 5] else range(10) # transformer_depth
#             for index in block_indices:
#                 _set_model_patch_replace(work_model, patch_kwargs_ip, ("input", id, index))
#                 patch_kwargs_ip["number"] += 1

#         for id in range(6): # id of output_blocks that have cross attention
#             block_indices = range(2) if id in [3, 4, 5] else range(10) # transformer_depth
#             for index in block_indices:
#                 _set_model_patch_replace(work_model, patch_kwargs_ip, ("output", id, index))
#                 patch_kwargs_ip["number"] += 1

#         for index in range(10):
#             _set_model_patch_replace(work_model, patch_kwargs_ip, ("middle", 0, index))
#             patch_kwargs_ip["number"] += 1

#         '''------------------------'''
#         # ip_weight = weight if ip_weight is None else ip_weight
#         # cn_strength = weight if cn_strength is None else cn_strength
#         work_model = work_model.clone()

#         face_embed = self.extract_face_emb(alkaid_loader['insightface'], image)
#         if face_embed is None:
#             raise Exception('Reference Image: No face detected.')

#         clip_embed = face_embed
#         # InstantID works better with averaged embeds (TODO: needs testing)

#         clip_embed = self.combine_embedding(type=combine_embeds, img_cond_embeds=clip_embed)

#         if noise > 0:
#             seed = int(torch.sum(clip_embed).item()) % 1000000007
#             torch.manual_seed(seed)
#             clip_embed_zeroed = noise * torch.rand_like(clip_embed)
#             #clip_embed_zeroed = add_noise(clip_embed, noise)
#         else:
#             clip_embed_zeroed = torch.zeros_like(clip_embed)

#         clip_embeddings_dim = face_embed.shape[-1]

#         self.instantid = InstantID(
#             adapterface_loader['instantid'],
#             cross_attention_dim=1280,
#             output_cross_attention_dim=output_cross_attention_dim,
#             clip_embeddings_dim=clip_embeddings_dim,
#             clip_extra_context_tokens=16,
#         )
#         print(f'clip_embeddings_dim: {clip_embeddings_dim}')


#         self.instantid.to(device, dtype=dtype)

#         image_prompt_embeds, uncond_image_prompt_embeds = self.instantid.get_image_embeds(clip_embed.to(device, dtype=dtype), clip_embed_zeroed.to(device, dtype=dtype))

#         image_prompt_embeds = image_prompt_embeds.to(device, dtype=dtype)
#         uncond_image_prompt_embeds = uncond_image_prompt_embeds.to(device, dtype=dtype)



#         patch_kwargs_id = {
#             "ipadapter": self.instantid,
#             "number": 0,
#             "weight": ip_weight,
#             "cond": image_prompt_embeds,
#             "uncond": uncond_image_prompt_embeds,
#             # "weight_type": weight_type,
#             # "cond": 0.7*image_prompt_embeds + 0.3*cond,
#             # "uncond": 0.7*uncond_image_prompt_embeds + 0.3*uncond,
#             "mask": mask,
#             "sigma_start": work_model.model.model_sampling.percent_to_sigma(0),
#             "sigma_end": work_model.model.model_sampling.percent_to_sigma(1),
#             "embeds_scaling": embeds_scaling_id,
#         }

#         for id in [4,5,7,8]: # id of input_blocks that have cross attention
#             block_indices = range(2) if id in [4, 5] else range(10) # transformer_depth
#             for index in block_indices:
#                 _set_model_patch_replace(work_model, patch_kwargs_id, ("input", id, index))
#                 patch_kwargs_id["number"] += 1

#         for id in range(6): # id of output_blocks that have cross attention
#             block_indices = range(2) if id in [3, 4, 5] else range(10) # transformer_depth
#             for index in block_indices:
#                 _set_model_patch_replace(work_model, patch_kwargs_id, ("output", id, index))
#                 patch_kwargs_id["number"] += 1
        
#         for index in range(10):
#             _set_model_patch_replace(work_model, patch_kwargs_id, ("middle", 0, index))
#             patch_kwargs_id["number"] += 1

#         '''------------------------'''
#         if mask is not None and len(mask.shape) < 3:
#             mask = mask.unsqueeze(0)
#         face_kps = image_kps[0:1]
#         cnets = {}
#         cond_uncond = []

#         is_cond = True
#         for conditioning in [positive, negative]:
#             c = []
#             for t in conditioning:
#                 d = t[1].copy()

#                 prev_cnet = d.get('control', None)
#                 if prev_cnet in cnets:
#                     c_net = cnets[prev_cnet]
#                 else:
#                     c_net = control_net.copy().set_cond_hint(face_kps.movedim(-1,1), cn_strength, (0, 1))
#                     c_net.set_previous_controlnet(prev_cnet)
#                     cnets[prev_cnet] = c_net

#                 d['control'] = c_net
#                 d['control_apply_to_uncond'] = False
#                 d['cross_attn_controlnet'] = image_prompt_embeds.to(comfy.model_management.intermediate_device()) if is_cond else uncond_image_prompt_embeds.to(comfy.model_management.intermediate_device())

#                 if mask is not None and is_cond:
#                     d['mask'] = mask
#                     d['set_area_to_bounds'] = False

#                 n = [t[0], d]
#                 c.append(n)
#             cond_uncond.append(c)
#             is_cond = False

#         return(work_model, cond_uncond[0], cond_uncond[1], )
    