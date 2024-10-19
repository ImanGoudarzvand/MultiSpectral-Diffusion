import os

import torch 
import matplotlib.pyplot as plt 
from pytorch_wavelets import DWTForward, DWTInverse


def transform_back(tensor, permute=False):
    if permute:
        return torch.clamp(((tensor + 1.) / 2. ).permute(1,2,0), 0, 1)
    return torch.clamp((tensor + 1.) / 2., 0, 1)

def take_wavelet(img_tensor):
    """Image tensor in [-1, +1] and [bs, in_chans, img_size, img_size]
    """
    dwt = DWTForward(J =1, wave="haar")
    low_subband, high_subbands = dwt(img_tensor) # [bs, c, h, w], [bs, c, 3, h, w]
    ll = low_subband / 2.
    lh, hl, _ = torch.unbind(high_subbands[0] / 2. , dim=2)

    return {"ll":ll, "lh":lh, "hl":hl} 

def save_wavelets_img(*subbands, name, path = None):
    """Subbands are ll, lh, hl, hh in order
    """
    assert len(subbands) == 4 , "all wavelets subband required"
    if not os.path.exists(path):
        os.makedirs(path)
    B = subbands[0].shape[0]
    path = "samples" if path is None else path
    for sample_i in range(B):
        _, ax = plt.subplots(2, 2,figsize=(4,4))
        for i in range(len(subbands)):
            ax[i // 2, i % 2].imshow(transform_back(subbands[i][sample_i], permute=True).cpu())
            ax[i // 2, i % 2].axis("off")
        plt.savefig(f"{path}/{name}_{sample_i}.jpg", format="jpg", dpi=300)
        plt.close()

def save_reconstructed_img(*subbands):
    img = take_wavelet_inverse(*subbands)
    img = ((img + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    img = img.permute(0, 2, 3, 1)
    img = img.contiguous()

    return img.numpy()  

def take_wavelet_inverse(*subbands):
    """Subbands are ll, lh, hl, hh 

    Returns:
        tensor image in [in_chans, img_size, img_size]
    """
    high_back = []
    idwt = DWTInverse('haar')
    assert len(subbands) == 4
    for subband in subbands:
        high_back.append(subband.unsqueeze(2))
    high_back = torch.concat(high_back, dim = 2) * 2.  
    high_back = high_back.float().cpu()
    constructed_img = idwt([high_back[:,:,0],[high_back[:,:,1:]]])
    return constructed_img