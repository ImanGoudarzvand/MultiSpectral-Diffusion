"""
Generate a large batch of video-audio pairs
"""
import sys,os
import argparse
import os
import numpy as np
import torch as th
import torch.distributed as dist
from .wavelet_util import save_wavelets_img, save_reconstructed_img
from . import dist_util, logger
from .script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict
)
from torch.utils.data import Dataset, DataLoader
from glob import glob 
from PIL import Image 
from torchvision.transforms import ToTensor, Compose, Resize, Lambda

class ImgDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        transform = None):

        super().__init__()
        self.data_dir = data_dir
        self.paths = sorted([p for p in glob(f"{data_dir}/*")], key = lambda x: int(x.split(".")[0].split("_")[-1]))
        self.transform = transform if transform else ToTensor()


    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        pil = Image.open(path)
        return self.transform(pil)
    
    
def load_data(data_dir,
            img_size,
            batch_size,
            num_workers=0):
    
    transform = Compose([
        Resize((img_size,img_size)),
        ToTensor(),
        Lambda(lambda x : x * 2 - 1)
        ])
    
    if not data_dir:
        raise ValueError("unspecified data directory")
    
    if not os.path.exists(data_dir):
        raise ValueError("data directory not found!")
    
    ds = ImgDataset(data_dir,
                    transform = transform)
    
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)
        
    return loader


def main():
    args = create_argparser().parse_args()
    
    dist_util.setup_dist()
    logger.configure(args.output_dir)

    if args.sample_fn != "ddpm":
        raise NotImplementedError(args.sample_fn)

    data_loader = load_data(args.data_dir, args.img_size//2 ,args.batch_size)


    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
         **args_to_dict(args, [key for key in model_and_diffusion_defaults().keys()])
    )

    if os.path.isdir(args.model_path):
        name_list = [model_name for model_name in os.listdir(args.model_path) \
            if (model_name.startswith('model') and model_name.endswith('.pt') and int(model_name.split('.')[0][5:])>= args.skip_steps)]
        name_list.sort()
        name_list = [os.path.join(args.model_path, model_name) for model_name in name_list[::1]]
    else:
        name_list = [model_path for model_path in args.model_path.split(',')]
        
    logger.log(f"models waiting to be evaluated:{name_list}")

    for model_path in name_list:
        model.load_state_dict(
            dist_util.load_state_dict(model_path, map_location=dist_util.dev()))
        
        model.to(dist_util.dev())
        if args.use_fp16:
            model.convert_to_fp16()
        model.eval()

        logger.log(f"sampling samples for {model_path}")
        model_name = model_path.split('/')[-1]

        wavelets_save_path = os.path.join(args.output_dir, model_name, 'wavelets')
        img_save_path = os.path.join(args.output_dir, model_name, 'upsampled_imgs')

        if dist.get_rank() == 0:
            os.makedirs(wavelets_save_path, exist_ok=True)

        upsampled_imgs = []

        for idx, data in enumerate(data_loader): 

            low_res_cond = {"ll": data.to(dist_util.dev())}

            shape = (args.batch_size , 3, args.image_size//2, args.image_size//2)


            sample_fn = diffusion.conditional_p_sample_loop


            sample = sample_fn(
                model,
                shape = shape,
                use_fp16= args.use_fp16,
                noise=None,
                clip_denoised=True,
                denoised_fn=None,
                cond_fn=None,
                model_kwargs=low_res_cond,
                device=None,
                progress=True
            )

            ll_samples = sample['ll']     # [num_samples, 3, img_size, img_size]
            lh_samples = sample['lh']     # [num_samples, 3, img_size, img_size]
            hl_samples = sample['hl']     # [num_samples, 3, img_size, img_size]

            save_wavelets_img(ll_samples.cpu(), lh_samples.cpu(), hl_samples.cpu(), th.zeros(args.batch_size, 3, args.image_size//2, args.image_size//2), name = f"subbands{idx}", path=wavelets_save_path)
            img_numpy = save_reconstructed_img(ll_samples.cpu(), lh_samples.cpu() , hl_samples.cpu() , th.zeros(args.batch_size, 3, args.image_size//2, args.image_size//2))
            upsampled_imgs.append(img_numpy)
            logger.log(f"upsampling {args.batch_size} images")
        
        np.savez(f"{img_save_path}", np.concatenate(upsampled_imgs, axis=0))

    logger.log("upsampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        batch_size=16,
        img_size = 64,
        data_dir="",
        model_path="",
        output_dir="upsampled",
        skip_steps = 1,
        num_workers=2,
        sample_fn="ddpm"

    )
   
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
