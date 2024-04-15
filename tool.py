import torch
import cv2
import numpy as np
import json

from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as TT
import torchvision.transforms.functional as TTF
import comfy.model_management
from comfyui_controlnet_aux.utils import common_annotator_call, create_node_input_types
import comfy.samplers
import comfy.utils
from nodes import common_ksampler

from .utils import (BBRegression, tensorToNP, tensor2pil, pil2tensor, draw_kps,
                    get_estimate_bbox, extract_5p)
from .ipadapter.utils import set_model_patch_replace


class FaceCrop:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "alkaid_loader": ("ALKAIDLOADER", ),
                "expand_ratio": ("FLOAT", {"default": 1.2, "min": 0.5, "max": 2.5, "step": 0.1}),
                "resized": ("INT", {"default": 768})
            },
        }

    RETURN_TYPES = ("IMAGE", "BBOXDETAIL")
    RETURN_NAMES = ("image_croped", "bbox detail")

    FUNCTION = "run_it"

    CATEGORY = "Alkaid/Tools"

    def get_bbox(self,image, insightface, expand_ratio):

        image = tensorToNP(image)
        face = insightface.get(image)
        h, w = image.shape[0], image.shape[1]
        if face.__len__() == 0:
            print('Detected no face, use all image')
            return np.array([0,0,w,h])

        face = sorted(face, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]
        
        kps = face['kps']
        bbox = BBRegression(np.array(kps).reshape([1, 10]))
        size = min(bbox[2], bbox[3])*expand_ratio
        center0 = bbox[0] + bbox[2]/2
        center1 = bbox[1] + bbox[3]/2
        if center0 < 0 or center1 < 0:
            print('Detected half face, use all image')
            return np.array([0,0,w,h])

        if center0 + size/2 > w or center0 - size/2 < 0:
            size = min(center0, w-center0)*2
            
        if center1 + size/2 > h or center1 - size/2 < 0:
            size = min(center1, h-center1)*2

        bbox = np.array([center0 - size/2, center1 - size/2, size, size]).astype(np.int32)

        bbox = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]

        bbox = np.array(bbox, dtype=np.int32)
        return bbox

    def run_it(self, image, alkaid_loader, resized, expand_ratio):
        insightface = alkaid_loader['insightface']

        # tensors = []
        image_crop_list = []
        bbox_list = []
        for img in image:
            bbox = self.get_bbox(img, insightface, expand_ratio)
            img = tensor2pil(img)
            img_crop = img.crop(bbox)
            img_crop = pil2tensor(img_crop.resize((resized, resized)) if resized != -1 else img_crop)

            image_crop_list.append(img_crop)
            bbox_list.append(bbox)
        croped = torch.concat(image_crop_list, dim=0)

        return (croped, bbox_list)


class FacePaste:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "image_croped": ("IMAGE",),
                "bbox_detail": ("BBOXDETAIL",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image")

    FUNCTION = "run_it"

    CATEGORY = "Alkaid/Tools"

    def run_it(self, image, image_croped, bbox_detail,):
        bs = image.shape[0]
        tensors = []
        for i in range(bs):
            img = image[i].cpu().numpy()
            img_copy = np.copy(img)
            img_croped = image_croped[i].cpu().numpy()
            
            bbox = bbox_detail[i]
            paste = cv2.resize(img_croped, (bbox[3]-bbox[1], bbox[2]-bbox[0]))
            img_copy[bbox[1]:bbox[3], bbox[0]:bbox[2]] = paste
            tensors.append(torch.from_numpy(img_copy)[None,...])

        tensors = torch.concat(tensors, dim=0)
        return (tensors,)


class OpenPose_Preprocessor:
    def __init__(self) -> None:
        from controlnet_aux.open_pose import OpenposeDetector
        self.model = OpenposeDetector.from_pretrained().to(comfy.model_management.get_torch_device())        


    @classmethod
    def INPUT_TYPES(s):
        return create_node_input_types(
            detect_hand = ("BOOLEAN", {"default": False}),
            detect_body = ("BOOLEAN", {"default": True}),
            detect_face = ("BOOLEAN", {"default": False}),
        )

        
    RETURN_TYPES = ("IMAGE", "POSE_KEYPOINT")
    FUNCTION = "estimate_pose"

    CATEGORY = "biubiubiu/Image"

    def estimate_pose(self, image, detect_hand, detect_body, detect_face, resolution=512, **kwargs):

        # detect_hand = detect_hand == "enable"
        # detect_body = detect_body == "enable"
        # detect_face = detect_face == "enable"

        self.openpose_dicts = []
        def func(image, **kwargs):
            pose_img, openpose_dict = self.model(image, **kwargs)

            self.openpose_dicts.append(openpose_dict)
            return pose_img
        out = common_annotator_call(func, image, include_hand=detect_hand, include_face=detect_face, include_body=detect_body, image_and_json=True, resolution=resolution)
        # del model
        return {
            'ui': { "openpose_json": [json.dumps(self.openpose_dicts, indent=4)] },
            "result": (out, self.openpose_dicts)
        }

def min_(tensor_list):
    # return the element-wise min of the tensor list.
    x = torch.stack(tensor_list)
    mn = x.min(axis=0)[0]
    return torch.clamp(mn, min=0)
    
def max_(tensor_list):
    # return the element-wise max of the tensor list.
    x = torch.stack(tensor_list)
    mx = x.max(axis=0)[0]
    return torch.clamp(mx, max=1)

class PrepImageForFace:

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE",),
            "alkaid_loader": ("ALKAIDLOADER", ),
            "type_string": ("STRING", {"default": ""}),

            "interpolation": (["LANCZOS", "BICUBIC", "HAMMING", "BILINEAR", "BOX", "NEAREST"],),
            "sharpening": ("FLOAT", {"default": 0.0, "min": 0, "max": 1, "step": 0.05}),
            "type_": (["face_center", "fusion", "resize"],),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "prep_image"

    CATEGORY = "Alkaid/Tools"

    @staticmethod
    def contrast_adaptive_sharpening(image, amount):
        img = F.pad(image, pad=(1, 1, 1, 1)).cpu()

        a = img[..., :-2, :-2]
        b = img[..., :-2, 1:-1]
        c = img[..., :-2, 2:]
        d = img[..., 1:-1, :-2]
        e = img[..., 1:-1, 1:-1]
        f = img[..., 1:-1, 2:]
        g = img[..., 2:, :-2]
        h = img[..., 2:, 1:-1]
        i = img[..., 2:, 2:]
        
        # Computing contrast
        cross = (b, d, e, f, h)
        mn = min_(cross)
        mx = max_(cross)
        
        diag = (a, c, g, i)
        mn2 = min_(diag)
        mx2 = max_(diag)
        mx = mx + mx2
        mn = mn + mn2
        
        # Computing local weight
        inv_mx = torch.reciprocal(mx)
        amp = inv_mx * torch.minimum(mn, (2 - mx))

        # scaling
        amp = torch.sqrt(amp)
        w = - amp * (amount * (1/5 - 1/8) + 1/8)
        div = torch.reciprocal(1 + 4*w)

        output = ((b + d + f + h)*w + e) * div
        output = torch.nan_to_num(output)
        output = output.clamp(0, 1)

        return (output)

    def get_bbox(self,image, insightface, expand_ratio=1.0):
        image = tensorToNP(image)
        face = insightface.get(image[0])
        h, w = image.shape[0], image.shape[1]
        if face.__len__() == 0:
            print('Detected no face, use all image')
            return np.array([0,0,w,h])

        face = sorted(face, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]
        
        kps = face['kps']
        bbox = BBRegression(np.array(kps).reshape([1, 10]))
        size = min(bbox[2], bbox[3])*expand_ratio
        center0 = bbox[0] + bbox[2]/2
        center1 = bbox[1] + bbox[3]/2
        if center0 < 0 or center1 < 0:
            print('Detected half face, use all image')
            return np.array([0,0,w,h])

        if center0 + size/2 > w or center0 - size/2 < 0:
            size = min(center0, w-center0)*2
            
        if center1 + size/2 > h or center1 - size/2 < 0:
            size = min(center1, h-center1)*2

        bbox = np.array([center0 - size/2, center1 - size/2, size, size]).astype(np.int32)
        

        bbox = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]

        bbox = np.array(bbox, dtype=np.int32)
        return bbox

    def crop_image(self, image, crop_position, interpolation="LANCZOS", size=(224,224), sharpening=0.0):
        _, oh, ow, _ = image.shape
        output = image.permute([0,3,1,2])

        crop_size = min(oh, ow)

        c1, c0 = crop_position
        if c0 - crop_size/2 < 0:
            y = 0
        else:
            if oh - c0 < crop_size/2:
                y = oh-crop_size
            else:
                y = int(c0 - crop_size/2)

        if c1 - crop_size/2 < 0:
            x = 0
        else:
            if ow - c1 < crop_size/2:
                x = ow-crop_size
            else:
                x = int(c1 - crop_size/2)
        
        x2 = x+crop_size
        y2 = y+crop_size

        # crop
        output = output[:, :, y:y2, x:x2]

        imgs = []
        for i in range(output.shape[0]):
            img = TT.ToPILImage()(output[i])
            img = img.resize(size, resample=Image.Resampling[interpolation])
            imgs.append(TT.ToTensor()(img))
        output = torch.stack(imgs, dim=0)
        imgs = None # zelous GC
        
        if sharpening > 0:
            output = self.contrast_adaptive_sharpening(output, sharpening)
        
        output = output.permute([0,2,3,1])
        return output

    def prep_image(self, image, alkaid_loader, type_string="", interpolation="LANCZOS",  sharpening=0.0, type_="center"):
        faceanalysis = alkaid_loader['insightface']
        if type_string == "":
            this_type = type_
        else:
            this_type = type_string
        size = (224, 224)
        imgs = []
        if this_type == "face_center":
            for i in range(image.shape[0]):
                img = image[i:i+1]
                face_bbox = self.get_bbox(img, faceanalysis)
                center_x = (face_bbox[0] + face_bbox[2])/2
                center_y = (face_bbox[1] + face_bbox[3])/2
                output_f = self.crop_image(image, (center_x, center_y), size=size, interpolation=interpolation, sharpening=sharpening)
                imgs.append(output_f)
        elif this_type == 'fusion':
            for i in range(image.shape[0]):
                img = image[i:i+1]
                # face_bbox = self.get_bbox(img, faceanalysis)
                # center_x = (face_bbox[0] + face_bbox[2])/2
                # center_y = (face_bbox[1] + face_bbox[3])/2
                # output_f = self.crop_image(image, (center_x, center_y), size=size, interpolation=interpolation, sharpening=sharpening)
                # imgs.append(output_f)

                _, oh, ow, _ = img.shape
                output0 = self.crop_image(image, (0, 0), size=size, interpolation=interpolation, sharpening=sharpening)
                output1 = self.crop_image(image, (oh-1, ow-1), size=size, interpolation=interpolation, sharpening=sharpening)
                imgs.append(output0)
                imgs.append(output1)

        elif this_type == "resize":
            for i in range(image.shape[0]):
                img = image[i].permute(2, 0, 1)
                img = TT.ToPILImage()(img)
                img = img.resize(size, resample=Image.Resampling[interpolation])
                img = TT.ToTensor()(img)[None, ...]
                imgs.append(img.permute(0, 2,3,1))
        else:
            return image

        
        output = torch.concat(imgs, dim=0)

        return (output, )
    
    
class Face3DSwapper:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "alkaid_loader": ("ALKAIDLOADER", ),
                "image_src": ("IMAGE", ),
                "image_template": ("IMAGE", ),
                
                "is_swapper": ("BOOLEAN", {"default": False})
            },
            "optional": {
                "dilate_size": ("INT", {"default": 33}),
                "blur_size": ("INT", {"default": 65}),
                "sigma": ("FLOAT", { "default": 10, "min": 0, "max": 100, "step": 0.1 }),

            }
        }
    RETURN_TYPES = ("IMAGE", 'MASK', "IMAGE")
    RETURN_NAMES = ("image", "mask", "verbose")

    FUNCTION = "run_it"
    CATEGORY = "Alkaid/Tools"

    def run_it(self, alkaid_loader, image_src, image_template, dilate_size=0, blur_size=0, sigma=1.5, is_swapper=False):

        insightface = alkaid_loader['insightface']
        estimate = alkaid_loader['estimate3d']
        image_src_crop_list = []

        for i in range(image_src.shape[0]):
            image_crop, _ = get_estimate_bbox(tensorToNP(image_src[i:i+1])[0], insightface=insightface)
            image_src_crop_list.append(image_crop[None, ...])
        image_crop_tensor = torch.from_numpy(np.concatenate(image_src_crop_list, axis=0)).float().permute(0, 3, 1, 2)
        

        template_crop, bbox = get_estimate_bbox(tensorToNP(image_template)[0], insightface=insightface)

        coeffs_list = estimate.estimate(image_crop_tensor/255)
        coeffs_id_mean = torch.concat([i['id'] for i in coeffs_list], dim=0).mean(0, keepdim=True)

        template_crop_tensor = torch.from_numpy(template_crop).float().unsqueeze(0).permute(0, 3, 1, 2)
        coeffs_template = estimate.estimate(template_crop_tensor/255)[0]
        if is_swapper:
            coeffs_template['id'] = coeffs_id_mean
            if coeffs_list.__len__() == 1:
                coeffs_template['exp'] = coeffs_list[0]['exp']

        landmarks, mask = estimate.coeffs_parse(coeffs_template)

        size = bbox[2] - bbox[0]

        landmarks = size*landmarks[0]/(estimate.center*2)
        landmarks[:, 0] += bbox[1]
        landmarks[:, 1] += bbox[0]

        h, w = image_template.shape[1], image_template.shape[2]

        kps_image = draw_kps(h,w, extract_5p(landmarks.cpu().numpy()))
        kps = pil2tensor(kps_image)

        

        mask = TTF.resize(mask, (size, size))
        mask_all = torch.zeros_like(image_template[..., 0])
        mask_all[:, bbox[0]:bbox[2], bbox[1]:bbox[3]] = mask

        mask_all = torch.clip(mask_all.to(torch.float32)/255, 0, 1)
        if dilate_size>0:
            mask_dilate =TTF.gaussian_blur(mask_all, kernel_size=dilate_size)
        else:
            mask_dilate = mask_all
        mask_dilate[mask_dilate!=0] = 1
        if blur_size > 0:
            mask_blur = TTF.gaussian_blur(mask_dilate, kernel_size=blur_size, sigma=sigma)
        else:
            mask_blur = mask_dilate
        mask_out = torch.clip(mask_blur, 0, 1)
        mask_out[mask_all==1] = 1
        return (kps, mask_out,  kps+image_template,)
    

class OpenPose_Preprocessor:
    def __init__(self) -> None:
        from controlnet_aux.open_pose import OpenposeDetector
        self.model = OpenposeDetector.from_pretrained().to(comfy.model_management.get_torch_device())        


    @classmethod
    def INPUT_TYPES(s):
        return create_node_input_types(
            detect_hand = ("BOOLEAN", {"default": False}),
            detect_body = ("BOOLEAN", {"default": True}),
            detect_face = ("BOOLEAN", {"default": False}),
        )

        
    RETURN_TYPES = ("IMAGE", "POSE_KEYPOINT")
    FUNCTION = "estimate_pose"

    CATEGORY = "Alkaid/Tools"

    def estimate_pose(self, image, detect_hand, detect_body, detect_face, resolution=512, **kwargs):

        # detect_hand = detect_hand == "enable"
        # detect_body = detect_body == "enable"
        # detect_face = detect_face == "enable"

        self.openpose_dicts = []
        def func(image, **kwargs):
            pose_img, openpose_dict = self.model(image, **kwargs)

            self.openpose_dicts.append(openpose_dict)
            return pose_img
        out = common_annotator_call(func, image, include_hand=detect_hand, include_face=detect_face, include_body=detect_body, image_and_json=True, resolution=resolution)
        # del model
        return {
            'ui': { "openpose_json": [json.dumps(self.openpose_dicts, indent=4)] },
            "result": (out, self.openpose_dicts)
        }
    

class CombineAdapterPatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "patch_1": ("ADAPTERPATCH", ),
                "patch_2": ("ADAPTERPATCH", ),

            },
            "optional": {
                "patch_3": ("ADAPTERPATCH", ),
                "patch_4": ("ADAPTERPATCH", ),
            }
        }
    RETURN_TYPES = ("ADAPTERPATCH",)
    RETURN_NAMES = ("adapter_patch", )
    FUNCTION = "combine"
    CATEGORY = "Alkaid/Adapter"

    def combine(self, patch_1, patch_2, patch_3=None, patch_4=None):
        patch = {}
        for p in [patch_1, patch_2, patch_3, patch_4,]:
            if p is None:
                continue
            for k in p.keys():
                if k in patch:
                    k_new = k + '_'
                else:
                    k_new = k
                patch[k_new] = p[k]
        return (patch, )


class ApplyAdapter:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "adapter_patch": ("ADAPTERPATCH", ),
            },
        }
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model", )
    FUNCTION = "apply"
    CATEGORY = "Alkaid/Adapter"

    def apply_patch(self, model, patch_kwargs):
        patch_kwargs["number"] = 0
        model = model.clone()
        for id in [4,5,7,8]: # id of input_blocks that have cross attention
            block_indices = range(2) if id in [4, 5] else range(10) # transformer_depth
            for index in block_indices:
                set_model_patch_replace(model, patch_kwargs, ("input", id, index))
                patch_kwargs["number"] += 1
        for id in range(6): # id of output_blocks that have cross attention
            block_indices = range(2) if id in [3, 4, 5] else range(10) # transformer_depth
            for index in block_indices:
                set_model_patch_replace(model, patch_kwargs, ("output", id, index))
                patch_kwargs["number"] += 1
        for index in range(10):
            set_model_patch_replace(model, patch_kwargs, ("middle", 0, index))
            patch_kwargs["number"] += 1
        return model

    def apply(self, model, adapter_patch):
        apply_keys = []
        work_model = model
        for k in ["style", 'ip', 'id',]:
            if k in adapter_patch:
                work_model = self.apply_patch(work_model, adapter_patch[k])
                apply_keys.append(k)
        for k in adapter_patch.keys():
            if k not in apply_keys:
                work_model = self.apply_patch(work_model, adapter_patch[k])
                apply_keys.append(k)

        return (work_model, )


class KSamplerHires:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "bislerp"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "add_noise": (["enable", "disable"], ),
                    "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 25, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "latent_image": ("LATENT", ),
                    "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                    "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                    "return_with_leftover_noise": (["disable", "enable"], ),
                    "is_hires": ("BOOLEAN", {"default": False}),
                     "upscale_method": (s.upscale_methods,),
                    "scale_by": ("FLOAT", {"default": 1.5, "min": 0.01, "max": 8.0, "step": 0.01}),

                    "steps_hires": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg_hires": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                    "sampler_name_hires": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler_hires": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "denoise_hires": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                     }
                }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"

    CATEGORY = "Alkaid/Sample"


    def upscale(self, samples, upscale_method, scale_by):
        s = samples.copy()
        width = round(samples["samples"].shape[3] * scale_by)
        height = round(samples["samples"].shape[2] * scale_by)
        s["samples"] = comfy.utils.common_upscale(samples["samples"], width, height, upscale_method, "disabled")
        return (s,)

    def sample(self, model, add_noise, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, start_at_step, end_at_step, return_with_leftover_noise,
               is_hires, upscale_method, scale_by, steps_hires, cfg_hires, sampler_name_hires, scheduler_hires, denoise_hires, denoise=1.0):
        force_full_denoise = True
        if return_with_leftover_noise == "enable":
            force_full_denoise = False
        disable_noise = False
        if add_noise == "disable":
            disable_noise = True
        latent_image, =  common_ksampler(model, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise, disable_noise=disable_noise, start_step=start_at_step, last_step=end_at_step, force_full_denoise=force_full_denoise)

        if is_hires:
            latent_image,  =  self.upscale(latent_image, upscale_method=upscale_method, scale_by=scale_by)
            latent_image, = common_ksampler(model, noise_seed, steps_hires, cfg_hires, sampler_name_hires, scheduler_hires, positive, negative, latent_image, denoise=denoise_hires)
        return (latent_image, )

class EmptyLatentImageLongside:
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "width": ("INT", {"default": 512, "min": 16, "max": 8192, "step": 8}),
            "height": ("INT", {"default": 512, "min": 16, "max": 8192, "step": 8}),
            
            "longside": ("INT", {"default": 1440}),
            "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096})}}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"

    CATEGORY = "Alkaid/Tools"

    def generate(self, width, height, longside, batch_size=1):
        longside_this = max(width, height)
        width = int(longside*(width/longside_this)//8)
        height = int(longside*(height/longside_this)//8)

        latent = torch.zeros([batch_size, 4, height, width], device=self.device)
        return ({"samples":latent}, )
    

class GetImageSize:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "execute"
    CATEGORY = "Alkaid/Tools"

    def execute(self, image):
        return (image.shape[2], image.shape[1],)