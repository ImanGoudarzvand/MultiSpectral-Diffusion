"""
Train a diffusion model on images.
"""

import argparse
import torch.distributed as dist

from . import dist_util, logger
from .image_datasets import load_data
from .resample import create_named_schedule_sampler
from .script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from .train_util import TrainLoop


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(args.output_dir, format_strs=["tensorboard", "csv", "log", "stdout"])

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)
    
    print("model entered")

    if dist.get_rank() == 0:
        logger.log("Effective parameters:")
        print("Effective parameters:")
        for key in sorted(args.__dict__):
            logger.log("  <<< {}: {}".format(key, args.__dict__[key]))
            print("  <<< {}: {}".format(key, args.__dict__[key]))

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        num_workers=args.num_workers
    )

    print("data entered")

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        initial_lg_loss_scale=args.initial_lg_loss_scale
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="data",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=2,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=4,
        save_interval=10000,
        output_dir="outputs",
        num_workers=0,
        resume_checkpoint="",
        initial_lg_loss_scale=20.0,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        multitask_loss_weights="0.7,0.15,0.15"
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
