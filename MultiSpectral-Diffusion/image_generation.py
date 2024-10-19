"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist
from .dpm_solver_plus import DPM_Solver  

from . import dist_util, logger
from .script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from .wavelet_util import save_reconstructed_img


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(args.output_dir)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("sampling...")

    all_ll = []
    all_lh = []
    all_hl = []

    group = 1

    all_labels = []
    while len(all_ll) * args.batch_size * args.num_gpus < args.num_samples: # 2: num gpus

        batch_ll = []
        batch_lh = []
        batch_hl = []

        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes

        shape = (args.batch_size , 3, 32, 32)
        
        if args.sample_fn == 'dpm_solver':

            dpm_solver = DPM_Solver(model=model, \
                alphas_cumprod=th.tensor(diffusion.alphas_cumprod, dtype=th.float32))
            
            x_T = {"ll":th.randn(*shape).to(dist_util.dev()), \
                "lh":th.randn(*shape).to(dist_util.dev()),
                "hl":th.randn(*shape).to(dist_util.dev())}
            
            sample = dpm_solver.sample(
                x_T,
                steps=20,
                order=2,
                skip_type="logSNR",
                method="adaptive",
            )

        elif args.sample_fn == 'dpm_solver++':
            dpm_solver = DPM_Solver(model=model, \
                alphas_cumprod=th.tensor(diffusion.alphas_cumprod, dtype=th.float32), \
                    predict_x0=True, thresholding=True)
            
            x_T = {"ll":th.randn(*shape).to(dist_util.dev()), \
                "lh":th.randn(*shape).to(dist_util.dev()),
                "hl":th.randn(*shape).to(dist_util.dev())}
            
            sample = dpm_solver.sample(
                x_T,
                steps=20,
                order=2,
                skip_type="logSNR",
                method="adaptive",
            )
        else:
            sample_fn = (
                diffusion.p_sample_loop if args.sample_fn=="ddpm" else diffusion.ddim_sample_loop
            )
            sample = sample_fn(
                model,
                (args.batch_size, 3, args.image_size // 2 , args.image_size // 2 ), # devide by 2 because we are sampling wavelet subbands
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                progress=False
            )

        gathered_sample_lls = [th.zeros_like(sample["ll"]) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_sample_lls, sample["ll"])  # gather not supported with NCCL
        batch_ll.extend([ll_sample.cpu() for ll_sample in gathered_sample_lls])

        gathered_sample_lhs = [th.zeros_like(sample["lh"]) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_sample_lhs, sample["lh"])  # gather not supported with NCCL
        batch_lh.extend([lh_sample.cpu() for lh_sample in gathered_sample_lhs])

        gathered_sample_hls = [th.zeros_like(sample["hl"]) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_sample_hls, sample["hl"])  # gather not supported with NCCL
        batch_hl.extend([hl_sample.cpu() for hl_sample in gathered_sample_hls])
        
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])


        if dist.get_rank() == 0:
            logger.log(f"created {len(batch_ll) * args.batch_size} samples")

        all_ll.append(th.cat(batch_ll, axis=0))
        all_lh.append(th.cat(batch_lh, axis=0))
        all_hl.append(th.cat(batch_hl, axis=0))


        group += 1

    ll_arr = th.concat(all_ll, axis=0)
    lh_arr = th.concat(all_lh, axis=0)
    hl_arr = th.concat(all_hl, axis=0)


    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]

    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in [ll_arr.shape[0], 3, args.image_size, args.image_size]])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        arr = save_reconstructed_img(ll_arr, lh_arr, hl_arr, th.zeros(ll_arr.shape[0], 3, args.image_size // 2 , args.image_size // 2))
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16, # per gpu
        model_path="",
        output_dir="samples",
        sample_fn="ddim",
        num_gpus = 1,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
