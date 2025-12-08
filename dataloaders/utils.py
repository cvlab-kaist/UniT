from typing import List, Dict
import random
import math
import os

import numpy as np
from PIL import Image
import cv2
import torch
from torch.nn import functional as F

import json 
import string

import glob

from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.data.transforms import paired_random_crop, triplet_random_crop
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt, random_add_speckle_noise_pt, random_add_saltpepper_noise_pt, bivariate_Gaussian


# unicode conversion: char <-> int
# use chr() and ord()
# char_table = [chr(i) for i in range(32,127)]
# valid_voc = list(string.printable[:-6])
# invalid_voc=['□', '∫', 'æ', '⬏', 'Σ', '■', 'Å', 'Ḏ', '£', 'ń', '⌀', 'Ù', '│', 'Ⅶ', 'Â', 'ς', 'Ⅻ', '⁴', 'ъ', '∁', 'Æ', 'α', 'Ç', 'ˣ', '・', '⤤', 'Đ', 'ı', '≡', '⋄', 'Å', 'ᴴ', 'ᵗ', 'Ȃ', 'δ', 'Ì', 'Ρ', '⟷', 'ï', '«', 'ȯ', 'Ǒ', '⇩', 'ζ', '✰', '⁹', 'м', 'Ộ', '❘', '₄', '²', 'φ', '⌴', '⇨', 'ƌ', 'σ', 'Ⅸ', '∞', 'ţ', 'ů', '◁', '½', '¾', 'ᴾ', '�', 'ê', 'Ⅵ', 'ˢ', '°', 'ɮ', '⇪', 'ᵈ', 'Ė', 'Ǐ', '⊲', '·', 'û', '˅', '⊤', '↰', 'Ī', 'ȍ', '×', '⊝', '‟', '√', '➀', 'î', '↹', '➞', '↑', 'ü', '⋏', '℃', 'Û', 'Ȅ', '›', '⟶', '○', 'Ⓡ', 'Ȋ', '➜', 'ᴺ', 'å', '►', '˂', 'ι', 'ā', 'Ś', '∇', '•', '¥', '★', '⋅', 'ₖ', 'ũ', '⁼', 'İ', '∓', '⊂', '➯', '₅', 'Ồ', '»', 'Ž', 'ì', 'Ⅴ', '„', 'Ň', 'ú', '‑', 'Ä', '⊣', '˄', '˙', 'Ó', '±', '╳', 'ⁿ', 'ū', 'ş', 'л', 'Ṡ', 'ᴵ', 'Ȏ', 'ñ', 'λ', '✓', 'ø', '✞', '≤', 'Õ', '⎯', '⬌', 'ʳ', 'Š', '◉', '➨', 'ᶜ', 'ź', 'ġ', 'ÿ', '◦', 'ḻ', '➮', 'ᴸ', 'Ú', '─', '⇧', '⤶', 'ð', 'ë', 'Ξ', 'ȑ', '⇦', '↻', 'ă', 'Ě', 'Ω', 'Á', '₃', 'к', 'Ⅰ', '▬', '—', '∈', 'Ạ', '☐', '⁸', 'Ŕ', 'ù', 'â', 'п', 'ᴭ', '÷', '↲', '‘', 'Ȇ', 'ᵀ', '¿', 'Ț', '▎', 'ě', 'ⱽ', 'Λ', '∷', '△', 'ç', 'ǫ', 'Ầ', '➩', 'и', 'Ū', 'ý', '―', '⇵', 'Í', 'ꝋ', '↓', '©', '³', 'Ɔ', 'è', '🠈', 'ğ', 'Ⓐ', 'я', 'Φ', 'Ấ', 'ᵖ', '︽', '˚', 'œ', '∥', 'β', 'й', 'Ⓒ', '⬍', '∨', '℮', '¼', 'ć', '␣', 'Ã', '🡨', 'Ą', 'ǵ', '™', 'Ế', 'ᵐ', '◄', 'Ń', '✱', 'ô', '¢', '₁', 'Ⅱ', '¹', 'π', 'µ', 'Ĺ', '⍙', 'р', 'Ï', 'ε', '⟵', '∆', 'ы', '⧫', 'ã', 'ė', '⁰', '⬉', '−', '⬋', '◯', 'о', 'À', 'ρ', '☰', 'τ', 'ŗ', '⸬', 'Ö', 'é', 'ə', 'Ǫ', 'Ē', '⎵', '𝔀', 'ⓒ', 'ȏ', '“', 'Č', 'č', 'Î', '∙', 'ṣ', '\u200b', '✚', 'ō', '”', 'ö', 'ᴹ', '▢', 'ν', '⌣', '：', '︾', '﹘', 'а', '∖', '⌄', 'в', '︿', 'ᵃ', 'ớ', '↺', '▲', '▽', '…', 'Ë', '⌫', '⤷', '€', '⊘', 'Ŏ', '₂', '⤺', '⁵', 'Ȧ', '∧', 'ω', '卐', 'Ⅳ', '⁻', '↵', 'ĩ', 'Ⅲ', 'Ă', '⬸', 'ʃ', 'ȇ', '←', '⅓', '⮌', '⇥', 'η', '➦', 'Ô', '⬊', '℉', '⊥', 'á', 'ŉ', '⊚', '–', 'Ā', '∅', 'Ć', '∎', '⤸', '⦁', 'ē', 'ί', 'õ', 'ᴱ', 'υ', 'ß', '◡', 'È', '∣', 'Δ', 'ᴙ', 'ò', '⊢', 'κ', '☓', 'Ề', 'Θ', 'ä', '﹀', '☆', 'Ò', '˃', 'à', 'Ê', 'ʰ', 'Ğ', '’', '→', '®', '●', '⁺', 'Ţ', 'Ż', '̓', '▼', 'Ể', 'ᵒ', 'Ý', 'б', '➔', 'г', '∴', '⅔', '⬈', 'Ō', '∊', 'Π', 'Ⅷ', 'Ñ', '➝', 'É', 'Ł', 'ó', '∉', 'Ø', 'Ü', '⋮', 'ĺ', '≣', '∼', '↱', 'í', 'Ⅹ', 'ę', '⋯', 'с', '╎', '⤦', '⊼', 'ȧ', '∝', '⤻', 'ξ', 'š', '▾', 'γ', '¡', '⊳', 'д', '⁷', 'ж', '➧', 'ᴰ', '‧', '∘', 'ž', 'Ȯ', 'Ⅺ']
CTLABELS = [' ','!','"','#','$','%','&','\'','(',')','*','+',',','-','.','/','0','1','2','3','4','5','6','7','8','9',':',';','<','=','>','?','@','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','[','\\',']','^','_','`','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','{','|','}','~']


def decode(idxs):
    s = ''
    for idx in idxs:
        if idx < len(CTLABELS):
            s += CTLABELS[idx]
        else:
            return s
    return s


def encode(word):
    s = []
    max_word_len = 25
    for i in range(max_word_len):
        if i < len(word):
            char=word[i]
            idx = CTLABELS.index(char)
            s.append(idx)
        else:
            s.append(96)
    return s


def load_data_files(opt, mode):

    files = []

    data_path = opt['hq_img_path']
    model_H, model_W = 512, 512 

    # load imgs
    hq_imgs_path = sorted(glob.glob(f'{data_path}/*.jpg'))
    
    # load anns
    ann_path = opt['ann_path']
    with open(ann_path, 'r') as f:
        anns = json.load(f)
        anns = sorted(anns.items())
    
    data_files = zip(hq_imgs_path, anns)
    for data_idx, (hq_img_path, ann) in enumerate(data_files):

        img_id = hq_img_path.split('/')[-1].split('.')[0]
        ann_id = ann[0]
        assert img_id == ann_id
        
        boxes=[]
        texts=[]
        text_encs=[]
        polys=[]

        
        # process anns
        img_anns = ann[1]['0']['text_instances']
        for img_ann in img_anns:

            # process text 
            text = img_ann['text']
            count=0
            for char in text:
                # only allow OCR english vocab: range(32,127)
                if 32 <= ord(char) and ord(char) < 127:
                    count+=1
                    # print(char, ord(char))
            if count == len(text) and count < 26:
                texts.append(text)
                text_encs.append(encode(text))
                assert text == decode(encode(text)), 'check text encoding !'
            else:
                continue

            # process box
            box_xyxy = img_ann['bbox']
            x1,y1,x2,y2 = box_xyxy
            box_xywh = [ x1, y1, x2-x1, y2-y1 ]
            box_xyxy_scaled = list(map(lambda x: x/model_H, box_xyxy))  # scale box coord to [0,1]
            x1,y1,x2,y2 = box_xyxy_scaled 
            box_cxcywh = [(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1]   # xyxy -> cxcywh
            # box format
            processed_box = box_cxcywh
            processed_box = list(map(lambda x: round(x,4), processed_box))
            boxes.append(processed_box)


            # process polygons
            poly = np.array(img_ann['polygon']).astype(np.int32)    # 16 2
            # scale poly
            poly_scaled = poly / np.array([model_W, model_H])
            polys.append(poly_scaled)

        assert len(boxes) == len(texts) == len(text_encs) == len(polys), f" Check loader!"

        # if the filetered image has no bbox and texts, skip it
        if len(boxes) == 0 or len(polys) == 0:
            continue

        files.append({  "img_path": hq_img_path, 
                        "text": texts, 
                        "bbox": boxes,
                        'poly': polys,
                        'text_enc': text_encs, 
                        "img_id": img_id})     

    return files


# https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/image_datasets.py
def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


# https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/image_datasets.py
def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


# https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/data/transforms.py
def augment(imgs, hflip=True, rotation=True, flows=None, return_status=False):
    """Augment: horizontal flips OR rotate (0, 90, 180, 270 degrees).

    We use vertical flip and transpose for rotation implementation.
    All the images in the list use the same augmentation.

    Args:
        imgs (list[ndarray] | ndarray): Images to be augmented. If the input
            is an ndarray, it will be transformed to a list.
        hflip (bool): Horizontal flip. Default: True.
        rotation (bool): Ratotation. Default: True.
        flows (list[ndarray]: Flows to be augmented. If the input is an
            ndarray, it will be transformed to a list.
            Dimension is (h, w, 2). Default: None.
        return_status (bool): Return the status of flip and rotation.
            Default: False.

    Returns:
        list[ndarray] | ndarray: Augmented images and flows. If returned
            results only have one element, just return ndarray.

    """
    hflip = hflip and random.random() < 0.5
    vflip = rotation and random.random() < 0.5
    rot90 = rotation and random.random() < 0.5

    def _augment(img):
        if hflip:  # horizontal
            cv2.flip(img, 1, img)
        if vflip:  # vertical
            cv2.flip(img, 0, img)
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    def _augment_flow(flow):
        if hflip:  # horizontal
            cv2.flip(flow, 1, flow)
            flow[:, :, 0] *= -1
        if vflip:  # vertical
            cv2.flip(flow, 0, flow)
            flow[:, :, 1] *= -1
        if rot90:
            flow = flow.transpose(1, 0, 2)
            flow = flow[:, :, [1, 0]]
        return flow

    if not isinstance(imgs, list):
        imgs = [imgs]
    imgs = [_augment(img) for img in imgs]
    if len(imgs) == 1:
        imgs = imgs[0]

    if flows is not None:
        if not isinstance(flows, list):
            flows = [flows]
        flows = [_augment_flow(flow) for flow in flows]
        if len(flows) == 1:
            flows = flows[0]
        return imgs, flows
    else:
        if return_status:
            return imgs, (hflip, vflip, rot90)
        else:
            return imgs


# https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/utils/img_process_util.py
def filter2D(img, kernel):
    """PyTorch version of cv2.filter2D

    Args:
        img (Tensor): (b, c, h, w)
        kernel (Tensor): (b, k, k)
    """
    k = kernel.size(-1)
    b, c, h, w = img.size()
    if k % 2 == 1:
        img = F.pad(img, (k // 2, k // 2, k // 2, k // 2), mode="reflect")
    else:
        raise ValueError("Wrong kernel size")

    ph, pw = img.size()[-2:]

    if kernel.size(0) == 1:
        # apply the same kernel to all batch images
        img = img.view(b * c, 1, ph, pw)
        kernel = kernel.view(1, 1, k, k)
        return F.conv2d(img, kernel, padding=0).view(b, c, h, w)
    else:
        # img: torch.Tensor
        img = img.view(1, b * c, ph, pw)
        kernel = kernel.view(b, 1, k, k).repeat(1, c, 1, 1).view(b * c, 1, k, k)
        return F.conv2d(img, kernel, groups=b * c).view(b, c, h, w)


class USMSharp(torch.nn.Module):

    def __init__(self, radius=50, sigma=0):
        super(USMSharp, self).__init__()
        if radius % 2 == 0:
            radius += 1
        self.radius = radius
        kernel = cv2.getGaussianKernel(radius, sigma)
        kernel = torch.FloatTensor(np.dot(kernel, kernel.transpose())).unsqueeze_(0)
        self.register_buffer("kernel", kernel)

    def forward(self, img, weight=0.5, threshold=10):
        blur = filter2D(img, self.kernel)
        residual = img - blur

        mask = torch.abs(residual) * 255 > threshold
        mask = mask.float()
        soft_mask = filter2D(mask, self.kernel)
        sharp = img + weight * residual
        sharp = torch.clip(sharp, 0, 1)
        return soft_mask * sharp + (1 - soft_mask) * img


def realesrgan_degradation(batch, use_usm=True, sf=4, resize_lq=True):

    #################### REALESRGAN SETTING ###########################
    args_degradation = {}
    # the first degradation process
    args_degradation['resize_prob'] = [0.2, 0.7, 0.1]  # up, down, keep
    args_degradation['resize_range'] = [0.15, 1.5]
    args_degradation['gaussian_noise_prob'] = 0.5
    args_degradation['noise_range'] = [1, 30]
    args_degradation['poisson_scale_range'] = [0.05, 3.0]
    args_degradation['gray_noise_prob'] = 0.4
    args_degradation['jpeg_range'] = [30, 95]

    # the second degradation process
    args_degradation['second_blur_prob'] = 0.8
    args_degradation['resize_prob2'] = [0.3, 0.4, 0.3]  # up, down, keep
    args_degradation['resize_range2'] = [0.3, 1.2]
    args_degradation['gaussian_noise_prob2'] = 0.5
    args_degradation['noise_range2'] = [1, 25]
    args_degradation['poisson_scale_range2'] = [0.05, 2.5]
    args_degradation['gray_noise_prob2'] = 0.4
    args_degradation['jpeg_range2'] = [30, 95]

    args_degradation['gt_size']= 512
    args_degradation['no_degradation_prob']= 0.01
    #################### REALESRGAN SETTING ###########################

    jpeger = DiffJPEG(differentiable=False).cuda()
    usm_sharpener = USMSharp().cuda()  # do usm sharpening
    im_gt = batch['gt'].cuda()
    if use_usm:
        im_gt = usm_sharpener(im_gt)
    im_gt = im_gt.to(memory_format=torch.contiguous_format).float()
    kernel1 = batch['kernel1'].cuda()
    kernel2 = batch['kernel2'].cuda()
    sinc_kernel = batch['sinc_kernel'].cuda()

    ori_h, ori_w = im_gt.size()[2:4]

    # ----------------------- The first degradation process ----------------------- #
    # blur
    out = filter2D(im_gt, kernel1)
    # random resize
    updown_type = random.choices(
            ['up', 'down', 'keep'],
            args_degradation['resize_prob'],
            )[0]
    if updown_type == 'up':
        scale = random.uniform(1, args_degradation['resize_range'][1])
    elif updown_type == 'down':
        scale = random.uniform(args_degradation['resize_range'][0], 1)
    else:
        scale = 1
    mode = random.choice(['area', 'bilinear', 'bicubic'])
    out = F.interpolate(out, scale_factor=scale, mode=mode)
    # add noise
    gray_noise_prob = args_degradation['gray_noise_prob']
    if random.random() < args_degradation['gaussian_noise_prob']:
        out = random_add_gaussian_noise_pt(
            out,
            sigma_range=args_degradation['noise_range'],
            clip=True,
            rounds=False,
            gray_prob=gray_noise_prob,
            )
    else:
        out = random_add_poisson_noise_pt(
            out,
            scale_range=args_degradation['poisson_scale_range'],
            gray_prob=gray_noise_prob,
            clip=True,
            rounds=False)
    # JPEG compression
    jpeg_p = out.new_zeros(out.size(0)).uniform_(*args_degradation['jpeg_range'])
    out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
    out = jpeger(out, quality=jpeg_p)

    # ----------------------- The second degradation process ----------------------- #
    # blur
    if random.random() < args_degradation['second_blur_prob']:
        out = filter2D(out, kernel2)
    # random resize
    updown_type = random.choices(
            ['up', 'down', 'keep'],
            args_degradation['resize_prob2'],
            )[0]
    if updown_type == 'up':
        scale = random.uniform(1, args_degradation['resize_range2'][1])
    elif updown_type == 'down':
        scale = random.uniform(args_degradation['resize_range2'][0], 1)
    else:
        scale = 1
    mode = random.choice(['area', 'bilinear', 'bicubic'])
    out = F.interpolate(
            out,
            size=(int(ori_h / sf * scale),
                    int(ori_w / sf * scale)),
            mode=mode,
            )
    # add noise
    gray_noise_prob = args_degradation['gray_noise_prob2']
    if random.random() < args_degradation['gaussian_noise_prob2']:
        out = random_add_gaussian_noise_pt(
            out,
            sigma_range=args_degradation['noise_range2'],
            clip=True,
            rounds=False,
            gray_prob=gray_noise_prob,
            )
    else:
        out = random_add_poisson_noise_pt(
            out,
            scale_range=args_degradation['poisson_scale_range2'],
            gray_prob=gray_noise_prob,
            clip=True,
            rounds=False,
            )

    # JPEG compression + the final sinc filter
    # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
    # as one operation.
    # We consider two orders:
    #   1. [resize back + sinc filter] + JPEG compression
    #   2. JPEG compression + [resize back + sinc filter]
    # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
    if random.random() < 0.5:
        # resize back + the final sinc filter
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(
                out,
                size=(ori_h // sf,
                        ori_w // sf),
                mode=mode,
                )
        out = filter2D(out, sinc_kernel)
        # JPEG compression
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*args_degradation['jpeg_range2'])
        out = torch.clamp(out, 0, 1)
        out = jpeger(out, quality=jpeg_p)
    else:
        # JPEG compression
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*args_degradation['jpeg_range2'])
        out = torch.clamp(out, 0, 1)
        out = jpeger(out, quality=jpeg_p)
        # resize back + the final sinc filter
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(
                out,
                size=(ori_h // sf,
                        ori_w // sf),
                mode=mode,
                )
        out = filter2D(out, sinc_kernel)

    # clamp and round
    im_lq = torch.clamp(out, 0, 1.0)

    # # random crop
    # gt_size = args_degradation['gt_size']
    # im_gt, im_lq = paired_random_crop(im_gt, im_lq, gt_size, sf)
    lq, gt = im_lq, im_gt

    # pho - resize lq to gt shape
    lq = F.interpolate(out, size=(im_gt.shape[-2], im_gt.shape[-1]), mode="bicubic")

    gt = torch.clamp(gt, 0, 1)
    lq = torch.clamp(lq, 0, 1)

    batch['gt'] = gt 
    batch['lq'] = lq
    return batch