from .tool import Face3DSwapper, FaceCrop, FacePaste, ApplyAdapter, CombineAdapterPatch, KSamplerHires
from .tool import OpenPose_Preprocessor, EmptyLatentImageLongside, GetImageSize
from .loader import AlkaidLoader, AdapterFaceLoader, AdapterStyleLoader
from .id_adapter import AdapterFace, ApplyControlNet_KPS

from .style_adapter import AdapterStyle

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']


NODE_CLASS_MAPPINGS= {

    "A_Face3DSwapper": Face3DSwapper,
    "A_FaceCrop": FaceCrop,
    "A_FacePaste": FacePaste,
    "A_OpenPosePreprocessor": OpenPose_Preprocessor,
    "A_EmptyLatentImageLongside": EmptyLatentImageLongside,
    "A_GetImageSize": GetImageSize,

    "AlkaidLoader": AlkaidLoader,
    "AdapterFaceLoader": AdapterFaceLoader,
    "AdapterStyleLoader": AdapterStyleLoader,

    "AdapterFace": AdapterFace,
    "AdapterStyle": AdapterStyle,

    "ApplyAdapter": ApplyAdapter,
    "ApplyControlNet_KPS": ApplyControlNet_KPS,

    "CombineAdapterPatch": CombineAdapterPatch,
    "KSamplerHires": KSamplerHires

}


NODE_DISPLAY_NAME_MAPPINGS = {
    "A_Face3DSwapper": 'A_Face3DSwapper',
    "A_FaceCrop": "A_FaceCrop",
    "A_FacePaste": "A_FacePaste",
    "A_OpenPosePreprocessor": "A_OpenPosePreprocessor",
    "A_EmptyLatentImageLongside": "A_EmptyLatentImageLongside",
    "A_GetImageSize": "A_GetImageSize",

    "AlkaidLoader": "AlkaidLoader",
    "AdapterFaceLoader": "AdapterFaceLoader",
    "AdapterStyleLoader": "AdapterStyleLoader",

    "AdapterFace": "AdapterFace",
    "AdapterStyle": "AdapterStyle",
    
    "ApplyAdapter": "ApplyAdapter",
    "ApplyControlNet_KPS": "ApplyControlNet_KPS",

    "CombineAdapterPatch": "CombineAdapterPatch",
    "KSamplerHires": "KSamplerHires"

}

