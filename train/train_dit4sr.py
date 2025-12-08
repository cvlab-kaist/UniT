import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
sys.path.append(os.getcwd())
from accelerate.logging import get_logger

import math
import wandb
import torch
import argparse
from tqdm.auto import tqdm
from einops import rearrange 
from omegaconf import OmegaConf

from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3, free_memory

import initialize 
from train_utils import encode_prompt, get_sigmas
from dataloaders.utils import realesrgan_degradation

logger = get_logger(__name__)

def main(cfg):

    
    # set experiment name
    exp_name = f'{cfg.train.mixed_precision}__{cfg.train.stage}__{cfg.log.tracker.msg}'
    
    
    # make save dir
    save_dir = f'{cfg.save.output_dir}/{exp_name}'
    os.makedirs(save_dir, exist_ok=True)

    
    # set accelerator and basic settings (seed, logging, dir_path)
    accelerator = initialize.load_experiment_setting(cfg, logger, exp_name)
    
    
    # set tracker
    initialize.load_trackers(cfg, accelerator, exp_name)


    # load train data
    train_dataloader = initialize.load_data(cfg)

    
    # load models 
    models = initialize.load_model(cfg, accelerator)


    # load model parameters (total_params, trainable_params, frozen_params)
    model_params = initialize.load_model_params(cfg, accelerator, models)


    # load optimizer 
    optimizer = initialize.load_optim(cfg, accelerator, models)


    # place models on cuda and proper weight dtype(float32, float16)
    weight_dtype = initialize.set_model_device(cfg, accelerator, models)


    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.train.gradient_accumulation_steps)
    if cfg.train.max_train_steps is None:
        cfg.train.max_train_steps = cfg.train.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        cfg.train.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.train.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=cfg.train.max_train_steps * accelerator.num_processes,
        num_cycles=cfg.train.lr_num_cycles,
        power=cfg.train.lr_power,
    )


    # Prepare everything with our `accelerator`.
    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(models['transformer'], optimizer, train_dataloader, lr_scheduler)


    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.train.gradient_accumulation_steps)
    if overrode_max_train_steps:
        cfg.train.max_train_steps = cfg.train.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    cfg.train.num_train_epochs = math.ceil(cfg.train.max_train_steps / num_update_steps_per_epoch)
    tot_train_epochs = cfg.train.num_train_epochs
    tot_train_steps = cfg.train.max_train_steps


    # Train!
    total_batch_size = cfg.train.batch_size * accelerator.num_processes * cfg.train.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info("=== Model Parameters ===")
    logger.info(f"  Total Params    : {model_params['tot_param']:,} ({model_params['tot_param']/1e6:.2f}M)")
    logger.info(f"  Trainable Params: {model_params['train_param']:,} ({model_params['train_param']/1e6:.2f}M)")
    logger.info(f"  Frozen Params   : {model_params['frozen_param']:,} ({model_params['frozen_param']/1e6:.2f}M)")

    logger.info("=== Training Setup ===")
    logger.info(f"  Num training samples = {len(train_dataloader.dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {tot_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {cfg.train.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {cfg.train.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {tot_train_steps}")

    
    # save trainable params as txt 
    if accelerator.is_main_process:
        txt_file = f'{save_dir}/train_params.txt'
        with open(txt_file, 'w') as f:
            # log trainable params 
            for name in model_params['train_param_names']:
                f.write(f'TRAINABLE - {name}\n')
            # log frozen params 
            for name in model_params['frozen_param_names']:
                f.write(f'FROZEN - {name}\n')


    global_step = 0
    first_epoch = 0
    initial_global_step = 0

    ocr_losses={}  

    progress_bar = tqdm(range(0, tot_train_steps), initial=initial_global_step, desc="Steps", disable=not accelerator.is_local_main_process,)
    free_memory()
    for epoch in range(first_epoch, tot_train_epochs):
        for step, batch in enumerate(train_dataloader):

            if cfg.data.train.name == 'satext':
                batch = realesrgan_degradation(batch)

                gt = batch['gt']
                lq = batch['lq']
                text = batch['text']
                text_encs = batch['text_enc']
                boxes = batch['bbox']    
                polys = batch['poly']    
                img_id = batch['img_id']
            

            with accelerator.accumulate([transformer]):

                if cfg.data.train.name == 'satext':
                    with torch.no_grad():
                        
                        # hq vae encoding
                        gt = gt.to(device=accelerator.device, dtype=weight_dtype) * 2.0 - 1.0   # b 3 512 512 
                        hq_latents = models['vae'].encode(gt).latent_dist.sample()  # b 16 64 64
                        model_input = (hq_latents - models['vae'].config.shift_factor) * models['vae'].config.scaling_factor    # b 16 64 64 
                        model_input = model_input.to(dtype=weight_dtype)
                        
                        # lq vae encoding
                        lq = lq.to(device=accelerator.device, dtype=weight_dtype) * 2.0 - 1.0   # b 3 512 512 
                        lq_latents = models['vae'].encode(lq).latent_dist.sample()  # b 16 64 64 
                        controlnet_image = (lq_latents - models['vae'].config.shift_factor) * models['vae'].config.scaling_factor   # b 16 64 64 
                        controlnet_image = controlnet_image.to(dtype=weight_dtype)

                        # gt text input for training
                        texts = [[f'"{t}"' for t in txt] for txt in text]
                        hq_prompt = [f'The image features the texts {", ".join(txt)} that appear clearly on signs, boards, buildings, or other objects.' for txt in texts]

                        # encode prompt 
                        prompt_embeds, pooled_prompt_embeds = encode_prompt(models['text_encoders'], models['tokenizers'], hq_prompt, 77)
                        prompt_embeds = prompt_embeds.to(model_input.dtype)                 # b 154 4096
                        pooled_prompt_embeds = pooled_prompt_embeds.to(model_input.dtype)   # b 2048
                else:
                    model_input = batch["pixel_values"].to(dtype=weight_dtype)
                    controlnet_image = batch["conditioning_pixel_values"].to(dtype=weight_dtype)


                # Sample noise that we'll add to the latents
                noise = torch.randn_like(model_input)   
                bsz = model_input.shape[0]
                # Sample a random timestep for each image
                # for weighting schemes where we sample timesteps non-uniformly
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=cfg.model.noise_scheduler.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=cfg.model.noise_scheduler.logit_mean,
                    logit_std=cfg.model.noise_scheduler.logit_std,
                    mode_scale=cfg.model.noise_scheduler.mode_scale,
                )

                indices = (u * models['noise_scheduler_copy'].config.num_train_timesteps).long()
                timesteps = models['noise_scheduler_copy'].timesteps[indices].to(device=model_input.device)

                # Add noise according to flow matching. b
                sigmas = get_sigmas(timesteps, accelerator, models['noise_scheduler_copy'], n_dim=model_input.ndim, dtype=model_input.dtype)   
                noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise   
                # Predict the noise residual
                trans_out = transformer(                       
                    hidden_states=noisy_model_input,            
                    controlnet_image=controlnet_image,          
                    timestep=timesteps,                         
                    encoder_hidden_states=prompt_embeds,        
                    pooled_projections=pooled_prompt_embeds,   
                    return_dict=False,
                    cfg=cfg
                )
                model_pred = trans_out[0]   

                if len(trans_out) > 1:
                    etc_out = trans_out[1]
                    # unpatchify
                    patch_size = models['transformer'].config.patch_size  
                    hidden_dim = models['transformer'].config.num_attention_heads * models['transformer'].config.attention_head_dim    
                    height = 64 // patch_size       
                    width = 64 // patch_size        
                    num_concat_feat = 2
                    extracted_feats = [ rearrange(feat['extract_feat'], 'b (N H W) (pH pW d) -> b (N d) (H pH) (W pW)', N=num_concat_feat, H=height, W=width, pH=patch_size, pW=patch_size) for feat in etc_out ]   

                # Follow: Section 5 of https://arxiv.org/abs/2206.00364.
                # Preconditioning of the model outputs.
                if cfg.model.noise_scheduler.precondition_outputs:   
                    model_pred = model_pred * (-sigmas) + noisy_model_input 

                # these weighting schemes use a uniform timestep sampling
                # and instead post-weight the loss
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=cfg.model.noise_scheduler.weighting_scheme, sigmas=sigmas)   

                # flow matching loss
                if cfg.model.noise_scheduler.precondition_outputs:   
                    target = model_input
                else:
                    target = noise - model_input

                # Compute regular diffusion loss.
                diff_loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                diff_loss = diff_loss.mean()

                # ts module loss 
                if 'ts_module' in cfg.train.model:
                    # process annotations for OCR training loss
                    train_targets=[]
                    for i in range(bsz):
                        num_box=len(boxes[i])
                        tmp_dict={}
                        tmp_dict['labels'] = torch.tensor([0]*num_box).to(accelerator.device)  # 0 for text
                        tmp_dict['boxes'] = torch.tensor(boxes[i]).to(accelerator.device)   # xyxy format, absolute coord, [num_box, 4]
                        tmp_dict['texts'] = text_encs[i]
                        tmp_dict['ctrl_points'] = polys[i]
                        train_targets.append(tmp_dict)

                    # OCR model forward pass
                    with torch.cuda.amp.autocast(enabled=False):
                        ocr_loss_dict, ocr_result = models['testr'](extracted_feats, train_targets, MODE='TRAIN')
                    # OCR loss
                    ocr_tot_loss = sum(v for v in ocr_loss_dict.values())
                    for ocr_key, ocr_val in ocr_loss_dict.items():
                        if ocr_key in ocr_losses.keys():
                            ocr_losses[ocr_key].append(ocr_val.item())
                        else:
                            ocr_losses[ocr_key]=[ocr_val.item()]

                
                # -------------------------------------------
                #   Loss calculation for different stages
                # -------------------------------------------
                if cfg.train.stage == 'stage1':
                    total_loss = diff_loss
                    ocr_tot_loss=torch.tensor(0).cuda()
                elif cfg.train.stage == 'stage2':
                    total_loss = cfg.train.ocr_loss_weight * ocr_tot_loss
                
                
                # backprop ! 
                if global_step > 0:
                    accelerator.backward(total_loss)
                    # clip gradients for stable training
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(list(transformer.parameters()), cfg.train.max_grad_norm)
                        if 'testr' in models and getattr(models['testr'], 'training', False):
                            torch.nn.utils.clip_grad_norm_(models['testr'].parameters(),
                                                        max_norm=cfg.train.max_grad_norm)
                    def get_max_grad(model):
                        max_grad = 0.0
                        for param in model.parameters():
                            if param.grad is not None:
                                max_grad = max(max_grad, param.grad.abs().max().item())
                        return max_grad
                    transformer_max_grad = get_max_grad(transformer)
                    testr_max_grad = get_max_grad(models['testr']) if 'testr' in models else 0.0

                    # -----------------------------
                    # WandB Logging
                    # -----------------------------
                    if accelerator.is_main_process and cfg.log.tracker.report_to == 'wandb':
                        wandb.log({
                            "gradients/transformer_max": transformer_max_grad,
                            "gradients/testr_max": testr_max_grad,
                            "loss/diff_loss": diff_loss.item(),
                            "loss/ocr_tot_loss": ocr_tot_loss.item(),
                            "loss/total_loss": total_loss.item(),
                        }, step=global_step)


                    # -----------------------------
                    # Optimizer step
                    # -----------------------------
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=cfg.train.set_grads_to_none)



            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # save model ckpt weight
                if accelerator.is_main_process:
                    if global_step % cfg.save.checkpointing_steps == 0:

                        # set save directory
                        save_path = f'{save_dir}/checkpoint-{global_step}'
                        os.makedirs(save_path, exist_ok=True)
                        
                        # save transformer
                        if 'transformer' in cfg.train.model:
                            accelerator.save_state(save_path)
                        
                        # save text spotting module 
                        if 'ts_module' in cfg.train.model:
                            # save ts_module
                            ts_ckpt = {}
                            ts_ckpt['ts_module'] = models['testr'].state_dict()
                            ckpt_path = f"{save_path}/ts_module{global_step:07d}.pt"
                            torch.save(ts_ckpt, ckpt_path)
                        logger.info(f"Saved state to {save_path}")
                        

            # log 
            logs = {"loss/total_loss": total_loss.detach().item(), 
                    'loss/diff_loss': diff_loss.detach().item(),
                    "optim/lr": lr_scheduler.get_last_lr()[0],
                    }
            
            # ocr log
            if 'ts_module' in cfg.train.model:
                logs["loss/ocr_tot_loss"] = ocr_tot_loss.detach().item()
                logs['optim/ts_module_lr'] = cfg.train.ts_module.lr
                for ocr_key, ocr_val in ocr_loss_dict.items():
                    logs[f"loss/ocr_{ocr_key}"] = ocr_val.detach().item()


            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= cfg.train.max_train_steps:
                break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UniT Training.")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    cfg.cfg_path = args.config
    if cfg.model.dit.resolution % 8 != 0:
        raise ValueError(
            "`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the controlnet encoder."
        )
    main(cfg)
