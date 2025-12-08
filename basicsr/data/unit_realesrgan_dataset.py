import cv2
import math
import numpy as np
import os
import os.path as osp
import random
import time
import torch
from pathlib import Path
from torch.utils import data as data
import glob

from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY



def is_non_flat(image, gradient_threshold=0.2):
    if image is None:
        return False
    
    # 计算梯度
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(sobelx**2 + sobely**2)
    
    # 计算梯度均值
    mean_gradient = np.mean(gradient)
    return mean_gradient > gradient_threshold

@DATASET_REGISTRY.register(suffix='basicsr')
class UniTRealESRGANDataset(data.Dataset):
    """Modified dataset based on the dataset used for Real-ESRGAN model:
    Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    It loads gt (Ground-Truth) images, and augments them.
    It also generates blur kernels and sinc kernels for generating low-quality images.
    Note that the low-quality images are processed in tensors on GPUS for faster processing.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            meta_info (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
            Please see more options in the codes.
    """

    def __init__(self, args, mode='train'):
        super(UniTRealESRGANDataset, self).__init__()

        opt = {}
        opt['data_name'] = args.name
        opt['hq_img_path'] = args.hq_img_path
        opt['ann_path'] = args.ann_path
        opt['null_text_ratio'] = args.null_text_ratio
        opt['queue_size'] = 160
        opt['crop_size'] =  512
        opt['io_backend'] = {}
        opt['io_backend']['type'] = 'disk'
        opt['blur_kernel_size'] = 21
        opt['kernel_list'] = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
        opt['kernel_prob'] = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
        opt['sinc_prob'] = 0.1
        opt['blur_sigma'] = [0.2, 3]
        opt['betag_range'] = [0.5, 4]
        opt['betap_range'] = [1, 2]
        opt['blur_kernel_size2'] = 11
        opt['kernel_list2'] = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
        opt['kernel_prob2'] = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
        opt['sinc_prob2'] = 0.1
        opt['blur_sigma2'] = [0.2, 1.5]
        opt['betag_range2'] = [0.5, 4.0]
        opt['betap_range2'] = [1, 2]
        opt['final_sinc_prob'] = 0.8
        opt['use_hflip'] = False
        opt['use_rot'] = False


        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        if 'crop_size' in opt:
            self.crop_size = opt['crop_size']
        else:
            self.crop_size = 512
        if 'image_type' not in opt:
            opt['image_type'] = 'png'
        # support multiple type of data: file path and meta data, remove support of lmdb
        self.paths = []
        if 'meta_info' in opt:
            with open(self.opt['meta_info']) as fin:
                    paths = [line.strip().split(' ')[0] for line in fin]
                    self.paths = [v for v in paths]
            if 'meta_num' in opt:
                self.paths = sorted(self.paths)[:opt['meta_num']]
        
        # pho
        if opt['data_name'] == 'satext':
            from dataloaders.utils import load_data_files
            self.paths = load_data_files(opt, mode)
            self.null_text_ratio = opt['null_text_ratio']

        if 'gt_path' in opt:    
            if isinstance(opt['gt_path'], str):
                self.paths.extend(sorted([str(x) for x in Path(opt['gt_path']).glob('*.'+opt['image_type'])]))
            else:
                self.paths.extend(sorted([str(x) for x in Path(opt['gt_path'][0]).glob('*.'+opt['image_type'])]))
                self.paths.extend(sorted([str(x) for x in Path(opt['gt_path'][0]).glob('*.'+'jpg')]))
                self.paths.extend(sorted([str(x) for x in Path(opt['gt_path'][0]).glob('*.'+'JPG')]))
                if len(opt['gt_path']) > 1:
                    for i in range(len(opt['gt_path'])-1):
                        self.paths.extend(sorted([str(x) for x in Path(opt['gt_path'][i+1]).glob('*.'+opt['image_type'])]))
                        self.paths.extend(sorted([str(x) for x in Path(opt['gt_path'][i+1]).glob('*.'+'jpg')]))
                        self.paths.extend(sorted([str(x) for x in Path(opt['gt_path'][i+1]).glob('*.'+'JPG')]))
        if 'imagenet_path' in opt:  # f
            class_list = os.listdir(opt['imagenet_path'])
            for class_file in class_list:
                self.paths.extend(sorted([str(x) for x in Path(os.path.join(opt['imagenet_path'], class_file)).glob('*.'+'JPEG')]))
        if 'face_gt_path' in opt:   # f
            if isinstance(opt['face_gt_path'], str):
                face_list = sorted([str(x) for x in Path(opt['face_gt_path']).glob('*.'+opt['image_type'])])
                self.paths.extend(face_list[:opt['num_face']])
            else:
                face_list = sorted([str(x) for x in Path(opt['face_gt_path'][0]).glob('*.'+opt['image_type'])])
                self.paths.extend(face_list[:opt['num_face']])
                if len(opt['face_gt_path']) > 1:
                    for i in range(len(opt['face_gt_path'])-1):
                        self.paths.extend(sorted([str(x) for x in Path(opt['face_gt_path'][0]).glob('*.'+opt['image_type'])])[:opt['num_face']])

        # limit number of pictures for test
        if 'num_pic' in opt:    # f
            if 'val' or 'test' in opt:
                random.shuffle(self.paths)
                self.paths = self.paths[:opt['num_pic']]
            else:
                self.paths = self.paths[:opt['num_pic']]

        if 'mul_num' in opt:     # f
            self.paths = self.paths * opt['mul_num']
            # print('>>>>>>>>>>>>>>>>>>>>>')
            # print(self.paths)

        # blur settings for the first degradation
        self.blur_kernel_size = opt['blur_kernel_size']
        self.kernel_list = opt['kernel_list']
        self.kernel_prob = opt['kernel_prob']  # a list for each kernel probability
        self.blur_sigma = opt['blur_sigma']
        self.betag_range = opt['betag_range']  # betag used in generalized Gaussian blur kernels
        self.betap_range = opt['betap_range']  # betap used in plateau blur kernels
        self.sinc_prob = opt['sinc_prob']  # the probability for sinc filters

        # blur settings for the second degradation
        self.blur_kernel_size2 = opt['blur_kernel_size2']
        self.kernel_list2 = opt['kernel_list2']
        self.kernel_prob2 = opt['kernel_prob2']
        self.blur_sigma2 = opt['blur_sigma2']
        self.betag_range2 = opt['betag_range2']
        self.betap_range2 = opt['betap_range2']
        self.sinc_prob2 = opt['sinc_prob2']

        # a final sinc filter
        self.final_sinc_prob = opt['final_sinc_prob']

        self.kernel_range = [2 * v + 1 for v in range(3, 11)]  # kernel size ranges from 7 to 21
        # TODO: kernel range is now hard-coded, should be in the configure file
        self.pulse_tensor = torch.zeros(21, 21).float()  # convolving with pulse tensor brings no blurry effect
        self.pulse_tensor[10, 10] = 1


    def __getitem__(self, index):


        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # -------------------------------- Load gt images -------------------------------- #
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
        data_file = self.paths[index]

        gt_path = data_file['img_path']
        text = data_file["text"]
        bbox = data_file["bbox"]
        text_enc = data_file["text_enc"]
        img_id = data_file['img_id']
        poly = data_file.get('poly')

        # avoid errors caused by high latency in reading files
        retry = 3
        while retry > 0:
            try:
                img_bytes = self.file_client.get(gt_path, 'gt')
            except (IOError, OSError) as e:
                # logger = get_root_logger()
                # logger.warn(f'File client error: {e}, remaining retry times: {retry - 1}')
                # change another file to read
                index = random.randint(0, self.__len__()-1)
                gt_path = self.paths[index]
                time.sleep(1)  # sleep 1s for occasional server congestion
            else:
                break
            finally:
                retry -= 1
        
        img_gt = imfrombytes(img_bytes, float32=True)   # orig_H, orig_W, 3 [0,1]

        if np.random.uniform() < self.null_text_ratio:
            hq_prompt = ""
        

        # -------------------- Do augmentation for training: flip, rotation -------------------- #
        img_gt = augment(img_gt, self.opt['use_hflip'], self.opt['use_rot'])

        # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.opt['sinc_prob']:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                kernel_size,
                self.blur_sigma,
                self.blur_sigma, [-math.pi, math.pi],
                self.betag_range,
                self.betap_range,
                noise_range=None)
        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.opt['sinc_prob2']:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(
                self.kernel_list2,
                self.kernel_prob2,
                kernel_size,
                self.blur_sigma2,
                self.blur_sigma2, [-math.pi, math.pi],
                self.betag_range2,
                self.betap_range2,
                noise_range=None)

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------------------- the final sinc kernel ------------------------------------- #
        if np.random.uniform() < self.opt['final_sinc_prob']:
            kernel_size = random.choice(self.kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.pulse_tensor

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt = img2tensor([img_gt], bgr2rgb=True, float32=True)[0]
        kernel = torch.FloatTensor(kernel)
        kernel2 = torch.FloatTensor(kernel2)

        return img_gt, kernel, kernel2, sinc_kernel, text, bbox, poly, text_enc, img_id


    def __len__(self):
        return len(self.paths)


def collate_fn_real(batch):

    gt, kernel, kernel2, sinc_kernel, text, bbox, poly, text_enc, img_id  = zip(*batch)
    # Convert lists of tensors to stacked tensors safely
    gt = torch.stack([x.clone().detach() for x in gt])
    kernel = torch.stack([x.clone().detach() for x in kernel])
    kernel2 = torch.stack([x.clone().detach() for x in kernel2])
    sinc_kernel = torch.stack([x.clone().detach() for x in sinc_kernel])
    
    text_enc_tensor=[]
    # preprocess text_enc
    for i in range(len(text_enc)):
        text_enc_tensor.append(torch.tensor(text_enc[i], dtype=torch.int32))

    poly_tensor=[]
    # process poly
    for i in range(len(poly)):
        poly_tensor.append(torch.tensor(np.array(poly[i]), dtype=torch.float32))
    
    return {
            "gt": gt,                           
            "kernel1": kernel,                  
            "kernel2": kernel2,
            "sinc_kernel": sinc_kernel,
            'text': list(text),
            'bbox': list(bbox),
            'poly': list(poly_tensor),
            'text_enc': list(text_enc_tensor),
            'img_id': list(img_id)
        }
