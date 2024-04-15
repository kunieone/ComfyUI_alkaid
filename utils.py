import os
import cv2
import numpy as np
import torch
import math
import PIL
from PIL import Image
import folder_paths


_BBREGRESSOR_PARAM_FILE = os.path.join(folder_paths.models_dir, "alkaid/BBRegressorParam.npy")

BBREGRESSOR_PARAM = None



def tensorToNP(image):
    out = torch.clamp(255. * image.detach().cpu(), 0, 255).to(torch.uint8)
    if out.shape[1] == 3:
        out = out.permute(0, 2, 3, 1)
    out = out.numpy()
    return out

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def BBRegression(points):
    global BBREGRESSOR_PARAM
    if BBREGRESSOR_PARAM is None:
        BBREGRESSOR_PARAM = np.load(_BBREGRESSOR_PARAM_FILE, allow_pickle=True).item()

    w1 = BBREGRESSOR_PARAM['W1']
    b1 = BBREGRESSOR_PARAM['B1']
    w2 = BBREGRESSOR_PARAM['W2']
    b2 = BBREGRESSOR_PARAM['B2']
    data = points.copy()
    data = data.reshape([5, 2])
    data_mean = np.mean(data, axis=0)
    x_mean = data_mean[0]
    y_mean = data_mean[1]
    data[:, 0] = data[:, 0] - x_mean
    data[:, 1] = data[:, 1] - y_mean

    rms = np.sqrt(np.sum(data ** 2) / 5)
    data = data / rms
    data = data.reshape([1, 10])
    data = np.transpose(data)
    inputs = np.matmul(w1, data) + b1
    inputs = 2 / (1 + np.exp(-2 * inputs)) - 1
    inputs = np.matmul(w2, inputs) + b2
    inputs = np.transpose(inputs)
    x = inputs[:, 0] * rms + x_mean
    y = inputs[:, 1] * rms + y_mean
    w = 224 / inputs[:, 2] * rms
    rects = [x, y, w, w]
    return np.array(rects).reshape([4])


def extract_5p(lm):
    lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1
    value_list = [
        lm[lm_idx[0], :],
        np.mean(lm[lm_idx[[1, 2]], :], 0),
        np.mean(lm[lm_idx[[3, 4]], :], 0), lm[lm_idx[5], :], lm[lm_idx[6], :]
    ]
    lm5p = np.stack(value_list, axis=0)
    lm5p = lm5p[[1, 2, 0, 3, 4], :]
    return lm5p


def get_estimate_bbox(image, insightface,  target_size=224., rescale_factor=102):
    face = insightface.get(image)
    h, w = image.shape[0], image.shape[1]
    if face.__len__() == 0:
        print('Detected no face, use all image')
        return image, np.array([0, 0, w, h])
    face = sorted(face, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]))[-1]
    kps = face['kps']
    global BBREGRESSOR_PARAM

    if BBREGRESSOR_PARAM is None:
        BBREGRESSOR_PARAM = np.load(_BBREGRESSOR_PARAM_FILE, allow_pickle=True).item()

    lm3d_std = BBREGRESSOR_PARAM['lm3d5_std']
    t, s = POS(kps.transpose(), lm3d_std.transpose())
    t = t.squeeze()
    s = rescale_factor / s

    l = int(t[0] - target_size / (2 * s))
    r = l + int(target_size / s)
    u = int(t[1] - target_size / (2 * s))
    b = u + int(target_size / s)
    bbox = np.array([u, l, b, r])
    image_crop = image[bbox[0]:bbox[2], bbox[1]:bbox[3]]
    image_crop = cv2.resize(image_crop, (224, 224))
    return image_crop, bbox


def POS(xp, x):
    npts = xp.shape[1]

    A = np.zeros([2 * npts, 8])

    A[0:2 * npts - 1:2, 0:3] = x.transpose()
    A[0:2 * npts - 1:2, 3] = 1

    A[1:2 * npts:2, 4:7] = x.transpose()
    A[1:2 * npts:2, 7] = 1

    b = np.reshape(xp.transpose(), [2 * npts, 1])

    k, _, _, _ = np.linalg.lstsq(A, b)

    R1 = k[0:3]
    R2 = k[4:7]
    sTx = k[3]
    sTy = k[7]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2)) / 2
    t = np.stack([sTx, sTy], axis=0)

    return t, s


def draw_kps(h, w, kps, color_list=[(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255)]):
    stickwidth = 4
    limbSeq = np.array([[0, 2], [1, 2], [3, 2], [4, 2]])
    kps = np.array(kps)

    out_img = np.zeros([h, w, 3])

    for i in range(len(limbSeq)):
        index = limbSeq[i]
        color = color_list[index[0]]

        x = kps[index][:, 0]
        y = kps[index][:, 1]
        length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(y[0] - y[1], x[0] - x[1]))
        polygon = cv2.ellipse2Poly((int(np.mean(x)), int(np.mean(y))), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        out_img = cv2.fillConvexPoly(out_img.copy(), polygon, color)
    out_img = (out_img * 0.6).astype(np.uint8)

    for idx_kp, kp in enumerate(kps):
        color = color_list[idx_kp]
        x, y = kp
        out_img = cv2.circle(out_img.copy(), (int(x), int(y)), 10, color, -1)

    out_img_pil = Image.fromarray(out_img.astype(np.uint8))
    return out_img_pil


def tensor_to_size(source, dest_size):
    if isinstance(dest_size, torch.Tensor):
        dest_size = dest_size.shape[0]
    source_size = source.shape[0]

    if source_size < dest_size:
        shape = [dest_size - source_size] + [1]*(source.dim()-1)
        source = torch.cat((source, source[-1:].repeat(shape)), dim=0)
    elif source_size > dest_size:
        source = source[:dest_size]

    return source

def tensor_to_image(tensor):
    image = tensor.mul(255).clamp(0, 255).byte().cpu()
    image = image[..., [2, 1, 0]].numpy()
    return image

def image_to_tensor(image):
    tensor = torch.clamp(torch.from_numpy(image).float() / 255., 0, 1)
    tensor = tensor[..., [2, 1, 0]]
    return tensor
