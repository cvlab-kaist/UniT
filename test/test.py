
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
sys.path.append(os.getcwd())

import glob
import torch
import argparse
from PIL import Image 
from omegaconf import OmegaConf
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.logging import get_logger

from pipelines.pipeline_unit import StableDiffusion3ControlNetPipeline
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

import initialize 
from test_utils import vlm_initial_text_extraction
from utils.wavelet_color_fix import adain_color_fix


logger = get_logger(__name__)


def main(cfg):
    
    
    # fix seed
    if cfg.init.seed is not None:
        set_seed(cfg.init.seed) 
    
    
    # set val saving dir
    os.makedirs(f'{cfg.save.output_dir}', exist_ok=True)
    
    
    # set accelerator
    accelerator = Accelerator(mixed_precision=cfg.train.mixed_precision)


    # load models 
    models = initialize.load_model(cfg, accelerator)
    
    
    # set cuda and proper dtype(float32, float16) 
    initialize.set_model_device(cfg, accelerator, models)
    
    
    # load vlm
    vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype=torch.bfloat16, device_map='auto')
    vlm_processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    vlm_model = vlm_model.eval()
    
    
    # load tsm 
    ts_module = models['testr'] 
        
        
    # load eval pipeline
    val_pipeline = StableDiffusion3ControlNetPipeline(
        vae=models['vae'], text_encoder=models['text_encoders'][0], text_encoder_2=models['text_encoders'][1], text_encoder_3=models['text_encoders'][2], 
        tokenizer=models['tokenizers'][0], tokenizer_2=models['tokenizers'][1], tokenizer_3=models['tokenizers'][2], 
        transformer=models['transformer'], scheduler=models['noise_scheduler'], ts_module=ts_module,
    )
    
    
    # load data
    lq_val_data = sorted(glob.glob(f'{cfg.data.val.eval_lq_path}/*.jpg'))


    # eval loop
    for val_lq_path in lq_val_data:
        
        # lq img info 
        val_img_id = val_lq_path.split('/')[-1].split('.')[0]
        
        
        # set seed 
        generator = None
        if accelerator.is_main_process and cfg.init.seed is not None:
            generator = torch.Generator(device=accelerator.device)
            generator.manual_seed(cfg.init.seed)


        # extract text from lq image 
        val_vlm_cap = vlm_initial_text_extraction(val_lq_path, vlm_model, vlm_processor)
        val_init_prompt = [val_vlm_cap]

        
        # process lq image 
        val_lq_pil = Image.open(val_lq_path).convert("RGB") # 128 128 
        ori_width, ori_height = val_lq_pil.size
        rscale = 4  # upscale x4
        # for shortest side smaller than 128, resize
        if ori_width < 512//rscale or ori_height < 512//rscale:
            scale = (512//rscale)/min(ori_width, ori_height)
            tmp_image = val_lq_pil.resize((int(scale*ori_width), int(scale*ori_height)),Image.BICUBIC)
            val_lq_pil = tmp_image
        val_lq_pil = val_lq_pil.resize((val_lq_pil.size[0]*rscale, val_lq_pil.size[1]*rscale), Image.BICUBIC)
        val_lq_pil = val_lq_pil.resize((val_lq_pil.size[0]//8*8, val_lq_pil.size[1]//8*8), Image.BICUBIC)
        
    
        # forward pass 
        with torch.no_grad():
            val_out = val_pipeline(
                prompt=val_init_prompt[0], control_image=val_lq_pil, num_inference_steps=cfg.data.val.num_inference_steps, generator=generator, height=512, width=512,
                guidance_scale=cfg.data.val.guidance_scale, negative_prompt=None,
                start_point=cfg.data.val.start_point, latent_tiled_size=cfg.data.val.latent_tiled_size, latent_tiled_overlap=cfg.data.val.latent_tiled_overlap,
                output_type = 'pil', return_dict=False, lq_id=val_img_id, cfg=cfg, vlm_model=vlm_model, vlm_processor=vlm_processor, val_lq_path=val_lq_path
            )
        
        
        # retrieve restored image 
        val_res_pil = val_out[0][0]  
        val_res_pil = adain_color_fix(val_res_pil, val_lq_pil)
        val_res_pil.save(f'{cfg.save.output_dir}/{val_img_id}.png')
        

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UniT")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    main(cfg)
