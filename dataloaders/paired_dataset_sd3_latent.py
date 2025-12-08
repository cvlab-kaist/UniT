import glob
import os
import random

import torch
from torchvision import transforms
from torch.utils import data as data


class PairedCaptionDataset(data.Dataset):
    def __init__(
            self,
            root_folder=None,
            null_text_ratio=0.2,
            # use_ram_encoder=False,
            # use_gt_caption=False,
            # caption_type = 'gt_caption',
    ):
        super(PairedCaptionDataset, self).__init__()

        # breakpoint()
        self.null_text_ratio = null_text_ratio
        self.lr_list = []
        self.gt_list = []
        # self.tag_path_list = []
        self.prompt_embeds_path_list = []
        self.pooled_prompt_embeds_path_list = []

        # for root_folder in root_folders:
        lr_path = root_folder +'/latent_lr'
        # tag_path = root_folder +'/tag'
        gt_path = root_folder +'/latent_hr'
        prompt_embeds_path = root_folder + '/prompt_embeds'
        pooled_prompt_embeds_path = root_folder + '/pooled_prompt_embeds'

        self.lr_list += glob.glob(os.path.join(lr_path, '*.pt'))
        self.gt_list += glob.glob(os.path.join(gt_path, '*.pt'))
        # self.tag_path_list += glob.glob(os.path.join(tag_path, '*.txt'))
        self.prompt_embeds_path_list += glob.glob(os.path.join(prompt_embeds_path, '*.pt'))
        self.pooled_prompt_embeds_path_list += glob.glob(os.path.join(pooled_prompt_embeds_path, '*.pt'))

        self.null_pooled_prompt_embeds_path = os.path.join(pooled_prompt_embeds_path, 'NULL_pooled_prompt_embeds.pt')
        self.null_prompt_embeds_path = os.path.join(prompt_embeds_path, 'NULL_prompt_embeds.pt')

        self.prompt_embeds_path_list.remove(self.null_prompt_embeds_path)
        self.pooled_prompt_embeds_path_list.remove(self.null_pooled_prompt_embeds_path)


        assert len(self.lr_list) == len(self.gt_list)
        assert len(self.lr_list) == len(self.prompt_embeds_path_list)

        self.img_preproc = transforms.Compose([       
            transforms.ToTensor(),
        ])


    def __getitem__(self, index):


        gt_path = self.gt_list[index]
        gt_latent = torch.load(gt_path)
        
        lq_path = self.lr_list[index]
        lq_latent = torch.load(lq_path)

        if random.random() < self.null_text_ratio:
            prompt_embeds = torch.load(self.null_prompt_embeds_path)
            pooled_prompt_embeds = torch.load(self.null_pooled_prompt_embeds_path)
        else:
            prompt_embeds_path = self.prompt_embeds_path_list[index]
            prompt_embeds = torch.load(prompt_embeds_path)

            pooled_prompt_embeds_path = self.pooled_prompt_embeds_path_list[index]
            pooled_prompt_embeds = torch.load(pooled_prompt_embeds_path)

        example = dict()
        example["conditioning_pixel_values"] = lq_latent.squeeze(0)
        example["pixel_values"] = gt_latent.squeeze(0)
        example['prompt_embeds'] = prompt_embeds.squeeze(0)
        example['pooled_prompt_embeds'] = pooled_prompt_embeds.squeeze(0)


        return example

    def __len__(self):
        return len(self.gt_list)