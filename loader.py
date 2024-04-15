import os
import re
import comfy.utils
import folder_paths
import comfy.model_management
from comfy.clip_vision import load as load_clip_vision
import torch
from .tools.estimate3dv import Estimate3DV
from .tools.faceskin import FaceSkin
from .ipadapter.init_proj import IDAdapter, StyleAdapter, InstantID


INSIGHTFACE_DIR = os.path.join(folder_paths.models_dir, "insightface")
ALKAID_DIR = os.path.join(folder_paths.models_dir, "alkaid")
IPADAPTER_DIR = os.path.join(folder_paths.models_dir, "ipadapter")

def insightface_loader(provider, name):
    try:
        from insightface.app import FaceAnalysis
    except ImportError as e:
        raise Exception(e)
    
    model = FaceAnalysis(name=name, root=INSIGHTFACE_DIR, providers=[provider + 'ExecutionProvider',], allowed_modules=['detection', 'recognition', ])
    model.prepare(ctx_id=0, det_size=(640, 640))
    return model



class AlkaidLoader:
    def __init__(self) -> None:
        self.insightface = None
        self.insightface_b = None
        self.estimate3d = None
        self.clip_vision = None

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            },
        }

    RETURN_TYPES = ("ALKAIDLOADER", )
    RETURN_NAMES = ("alkaid_loader", )
    FUNCTION = "load_models"
    CATEGORY = "Alkaid/Loader"

    @staticmethod
    def get_clipvision():
        clipvision_list = folder_paths.get_filename_list("clip_vision")


        pattern = '(ViT.H.14.*s32B.b79K.(bin|safetensors))'
        clipvision_file = [e for e in clipvision_list if re.search(pattern, e, re.IGNORECASE)]

        clipvision_file = folder_paths.get_full_path("clip_vision", clipvision_file[0]) if clipvision_file else None
        clip_vision = comfy.clip_vision.load(clipvision_file)
        comfy.model_management.load_model_gpu(clip_vision.patcher)
        return clip_vision


    def load_models(self,):
        device = comfy.model_management.get_torch_device()
        provider = "CUDA"
        self.insightface = insightface_loader(provider, name='antelopev2')
        self.insightface_b = insightface_loader(provider, name='buffalo_l')
        estimate3d = Estimate3DV(checkpoint=os.path.join(ALKAID_DIR, 'bfm_estimate.pth')).eval()
        self.estimate3d = estimate3d.to(device=device, )
        self.clip_vision = self.get_clipvision()
        return ({'insightface': self.insightface,
                 'insightface_b': self.insightface_b,
                 'estimate3d': self.estimate3d,
                 'clipvision': self.clip_vision
                 },)
    

# class AdapterFaceLoader_:

#     @classmethod
#     def INPUT_TYPES(s):
#         return {"required": {
#         }}

#     RETURN_TYPES = ("ADAPTERFACELOADER",)
#     RETURN_NAMES = ("adapterface_loader", )

#     FUNCTION = "load_model"
#     CATEGORY = "Alkaid/Loader"

#     def load_model(self):
#         device = comfy.model_management.get_torch_device()
#         dtype = torch.float16 if comfy.model_management.should_use_fp16() else torch.bfloat16 if comfy.model_management.should_use_bf16() else torch.float32

#         ckpt_path = os.path.join(ALKAID_DIR, "idadapter.bin")
#         model = comfy.utils.load_torch_file(ckpt_path, safe_load=True)


#         return (model, )
    

class AdapterFaceLoader:

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
        }}

    RETURN_TYPES = ("ADAPTERFACELOADER",)
    RETURN_NAMES = ("adapterface_loader", )

    FUNCTION = "load_model"
    CATEGORY = "Alkaid/Loader"

    def load_model(self):
        device = comfy.model_management.get_torch_device()
        dtype = torch.float16 if comfy.model_management.should_use_fp16() else torch.bfloat16 if comfy.model_management.should_use_bf16() else torch.float32

        ckpt_path = os.path.join(ALKAID_DIR, "idadapter.bin")
        model = comfy.utils.load_torch_file(ckpt_path, safe_load=True)
        output_cross_attention_dim = 2048

        idadapter = IDAdapter(
            model['ipadapter'],
            cross_attention_dim=2048,
            output_cross_attention_dim=output_cross_attention_dim,
            clip_embeddings_dim=512,
            clip_extra_context_tokens=16,
        ).to(dtype=dtype)

        instantid = InstantID(
            model['instantid'],
            cross_attention_dim=1280,
            output_cross_attention_dim=output_cross_attention_dim,
            clip_embeddings_dim=512,
            clip_extra_context_tokens=16,
        ).to(dtype=dtype)

        return (
            {
                'ipadapter': idadapter,
                'instantid': instantid
            }, 
        )

class AdapterStyleLoader:

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
        }}

    RETURN_TYPES = ("ADAPTERSTYLELOADER",)
    RETURN_NAMES = ("adapterstyle_loader", )

    FUNCTION = "load_model"
    CATEGORY = "Alkaid/Loader"

    def load_model(self):
        device = comfy.model_management.get_torch_device()
        dtype = torch.float16 if comfy.model_management.should_use_fp16() else torch.bfloat16 if comfy.model_management.should_use_bf16() else torch.float32

        ckpt_path = os.path.join(ALKAID_DIR, "style_xl.safetensors")
        model = comfy.utils.load_torch_file(ckpt_path, safe_load=True)
        if ckpt_path.lower().endswith(".safetensors"):
            st_model = {"image_proj": {}, "ip_adapter": {}}
            for key in model.keys():
                if key.startswith("image_proj."):
                    st_model["image_proj"][key.replace("image_proj.", "")] = model[key]
                elif key.startswith("ip_adapter."):
                    st_model["ip_adapter"][key.replace("ip_adapter.", "")] = model[key]
            model = st_model
            del st_model


        output_cross_attention_dim = 2048

        ipa = StyleAdapter(
            model,
            cross_attention_dim=1280,
            output_cross_attention_dim=output_cross_attention_dim,
            clip_embeddings_dim=1280,
            clip_extra_context_tokens=16,
        ).to(dtype=dtype)

        return ({
            "adapter": ipa
        }, )
