from typing import List, Optional
from pathlib import Path
from models import utils as mutils
from sde_lib import VESDE
from sampling import (ReverseDiffusionPredictor,
                      LangevinCorrector,
                      get_pc_fouriercs_fast)
from models import ncsnpp
import time
from utils import fft2, ifft2, get_mask, get_data_scaler, get_data_inverse_scaler, restore_checkpoint
import torch
import torch.nn as nn
import numpy as np
from models.ema import ExponentialMovingAverage
import matplotlib.pyplot as plt
import importlib
import argparse
import h5py
import torchvision.transforms as T
import torchvision
import imageio

def fft2c_new(data: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    """
    Apply centered 2 dimensional Fast Fourier Transform.

    Args:
        data: Complex valued input data containing at least 3 dimensions:
            dimensions -3 & -2 are spatial dimensions and dimension -1 has size
            2. All other dimensions are assumed to be batch dimensions.
        norm: Normalization mode. See ``torch.fft.fft``.

    Returns:
        The FFT of the input.
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    data = ifftshift(data, dim=[-3, -2])
    data = torch.view_as_real(
        torch.fft.fftn(  # type: ignore
            torch.view_as_complex(data), dim=(-2, -1), norm=norm
        )
    )
    data = fftshift(data, dim=[-3, -2])

    return data


def ifft2c_new(data: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    """
    Apply centered 2-dimensional Inverse Fast Fourier Transform.

    Args:
        data: Complex valued input data containing at least 3 dimensions:
            dimensions -3 & -2 are spatial dimensions and dimension -1 has size
            2. All other dimensions are assumed to be batch dimensions.
        norm: Normalization mode. See ``torch.fft.ifft``.

    Returns:
        The IFFT of the input.
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    data = ifftshift(data, dim=[-3, -2])
    data = torch.view_as_real(
        torch.fft.ifftn(  # type: ignore
            torch.view_as_complex(data), dim=(-2, -1), norm=norm
        )
    )
    data = fftshift(data, dim=[-3, -2])

    return data


# Helper functions


def roll_one_dim(x: torch.Tensor, shift: int, dim: int) -> torch.Tensor:
    """
    Similar to roll but for only one dim.

    Args:
        x: A PyTorch tensor.
        shift: Amount to roll.
        dim: Which dimension to roll.

    Returns:
        Rolled version of x.
    """
    shift = shift % x.size(dim)
    if shift == 0:
        return x

    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)

    return torch.cat((right, left), dim=dim)


def roll(
    x: torch.Tensor,
    shift: List[int],
    dim: List[int],
) -> torch.Tensor:
    """
    Similar to np.roll but applies to PyTorch Tensors.

    Args:
        x: A PyTorch tensor.
        shift: Amount to roll.
        dim: Which dimension to roll.

    Returns:
        Rolled version of x.
    """
    if len(shift) != len(dim):
        raise ValueError("len(shift) must match len(dim)")

    for (s, d) in zip(shift, dim):
        x = roll_one_dim(x, s, d)

    return x


def fftshift(x: torch.Tensor, dim: Optional[List[int]] = None) -> torch.Tensor:
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors

    Args:
        x: A PyTorch tensor.
        dim: Which dimension to fftshift.

    Returns:
        fftshifted version of x.
    """
    if dim is None:
        # this weird code is necessary for toch.jit.script typing
        dim = [0] * (x.dim())
        for i in range(1, x.dim()):
            dim[i] = i

    # also necessary for torch.jit.script
    shift = [0] * len(dim)
    for i, dim_num in enumerate(dim):
        shift[i] = x.shape[dim_num] // 2

    return roll(x, shift, dim)


def ifftshift(x: torch.Tensor, dim: Optional[List[int]] = None) -> torch.Tensor:
    """
    Similar to np.fft.ifftshift but applies to PyTorch Tensors

    Args:
        x: A PyTorch tensor.
        dim: Which dimension to ifftshift.

    Returns:
        ifftshifted version of x.
    """
    if dim is None:
        # this weird code is necessary for toch.jit.script typing
        dim = [0] * (x.dim())
        for i in range(1, x.dim()):
            dim[i] = i

    # also necessary for torch.jit.script
    shift = [0] * len(dim)
    for i, dim_num in enumerate(dim):
        shift[i] = (x.shape[dim_num] + 1) // 2

    return roll(x, shift, dim)


def main():
    ###############################################
    # 1. Configurations
    ###############################################

    # args
    args = create_argparser().parse_args()
    N = args.N
    m = args.m
    # fname = '001'
    fname = args.data
    filename = f'./samples/real/{args.task}/{fname}.npy'
    mask_filename = f'./samples/real/prospective/{fname}_mask.npy'

    print('initializing...')
    configs = importlib.import_module(f"configs.ve.fastmri_knee_320_ncsnpp_continuous")
    config = configs.get_config()
    img_size = config.data.image_size
    batch_size = 1

    # Read data
    img = torch.from_numpy(np.load(filename))
    print(img.type())
    print(img.shape)

    plt.imshow(np.log(np.abs(img)), cmap='gray')
    plt.savefig('their_fftinputnp.png')
    
    file = "/vol/datasets/cil/2021_11_23_fastMRI_data/knee/unzipped/singlecoil_train/file1000898.h5"
    hf = h5py.File(file, 'r')
    print(hf['kspace'].shape)

    k=hf['kspace'][0,hf['kspace'].shape[1]//2,:,:]
    plt.imshow(np.log(np.abs(k)), cmap='gray')
    plt.savefig('fftinputnp.png')

    #torchvision.utils.save_image(np.log(np.abs(k.astype(np.float32))), 'fftnpinput.png')
    print(k.dtype)
    #kt = torch.from_numpy(k)
    kt = torch.from_numpy(np.stack((k.real, k.imag), axis=-1))
    print(kt.dtype)
    #torchvision.utils.save_image(torch.log(kt.real), 'fftinput.png')
    plt.imshow(torch.log(torch.abs(kt[:,:,0])).numpy(), cmap='gray')
    plt.savefig('fftinput_.png')
    ifftimg = ifft2c_new(kt)#torch.fft.ifft2(torch.fft.ifftshift(kt))#.unsqueeze(0)
    print(ifftimg.shape)
    #transform = T.Resize((320,320))
    #fftimg = transform(fftimg).squeeze(0)
    cropx = (hf['kspace'].shape[2]-320)//2
    cropy = (hf['kspace'].shape[3]-320)//2 
    ifftimg=ifftimg[cropx:cropx+320,cropy:cropy+320,:]
    print(ifftimg.type())
    print(ifftimg.shape)
    plt.imshow(torch.abs(ifftimg[:,:,0]).numpy(), cmap='gray')
    plt.savefig('ifftinput_.png')
    img = ifftimg[:,:,0]
    #torchvision.utils.save_image(torch.abs(fftimg)*255, 'ifftinput.png')
    '''
    img = fft2c_new(ifftimg)#torch.fft.fft2(torch.fft.fftshift(fftimg)) 
    print(img.shape)
    plt.clf()
    plt.imshow(torch.log(torch.abs(img[:,:,0])).numpy(), cmap='gray')
    plt.savefig('finalfftinput_.png')
    print(img.type())

    img = img[:,:,0]
    print(img.type())
    print(img.shape)
    '''
    #cropx = (hf['kspace'].shape[2]-320)//2
    #cropy = (hf['kspace'].shape[3]-320)//2
    #k=hf['kspace'][0,hf['kspace'].shape[1]//2,cropx:cropx+320,cropy:cropy+320]
    #print(k.shape)
    #img = torch.from_numpy(k)
    print(img.shape)
    img = img.view(1, 1, 320, 320)
    img = img.to(config.device)
    if args.task == 'retrospective':
        # generate mask
        mask = get_mask(img, img_size, batch_size,
                        type=args.mask_type,
                        acc_factor=args.acc_factor,
                        center_fraction=args.center_fraction)
    elif args.task == 'prospective':
        mask = torch.from_numpy(np.load(mask_filename))
        mask = mask.view(1, 1, 320, 320)

    ckpt_filename = f"./weights/checkpoint_95.pth"
    sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=N)
    
    config.training.batch_size = batch_size
    predictor = ReverseDiffusionPredictor
    corrector = LangevinCorrector
    probability_flow = False
    snr = 0.16

    # sigmas = mutils.get_sigmas(config)
    scaler = get_data_scaler(config)
    inverse_scaler = get_data_inverse_scaler(config)

    # create model and load checkpoint
    score_model = mutils.create_model(config)
    ema = ExponentialMovingAverage(score_model.parameters(),
                                   decay=config.model.ema_rate)
    state = dict(step=0, model=score_model, ema=ema)
    state = restore_checkpoint(ckpt_filename, state, config.device, skip_sigma=True)
    ema.copy_to(score_model.parameters())

    # Specify save directory for saving generated samples
    save_root = Path(f'./results/real')
    save_root.mkdir(parents=True, exist_ok=True)

    irl_types = ['input', 'recon', 'recon_progress', 'label']
    for t in irl_types:
        save_root_f = save_root / t
        save_root_f.mkdir(parents=True, exist_ok=True)

    ###############################################
    # 2. Inference
    ###############################################

    pc_fouriercs = get_pc_fouriercs_fast(sde,
                                         predictor, corrector,
                                         inverse_scaler,
                                         snr=snr,
                                         n_steps=m,
                                         probability_flow=probability_flow,
                                         continuous=config.training.continuous,
                                         denoise=True,
                                         save_progress=True,
                                         save_root=save_root / 'recon_progress')
    # fft
    kspace = fft2(img)
    #kspace = img
    
    # undersampling
    under_kspace = kspace * mask
    under_img = torch.real(ifft2(under_kspace))

    print(f'Beginning inference')
    tic = time.time()
    x = pc_fouriercs(score_model, scaler(under_img), mask, Fy=under_kspace)
    toc = time.time() - tic
    print(f'Time took for recon: {toc} secs.')

    ###############################################
    # 3. Saving recon
    ###############################################

    if args.task == 'retrospective':
        # save input and label only if this is retrospective recon.
        input = under_img.squeeze().cpu().detach().numpy()
        label = img.squeeze().cpu().detach().numpy()
        mask_sv = mask.squeeze().cpu().detach().numpy()

        np.save(str(save_root / 'input' / fname) + '.npy', input)
        np.save(str(save_root / 'input' / (fname + '_mask')) + '.npy', mask_sv)
        np.save(str(save_root / 'label' / fname) + '.npy', label)
        plt.imsave(str(save_root / 'input' / fname) + '.png', input, cmap='gray')
        plt.imsave(str(save_root / 'label' / fname) + '.png', label, cmap='gray')

    recon = x.squeeze().cpu().detach().numpy()
    np.save(str(save_root / 'recon' / fname) + '.npy', recon)
    plt.imsave(str(save_root / 'recon' / fname) + '.png', recon, cmap='gray')


def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['retrospective', 'prospective'], default='retrospective',
                        type=str, help='If retrospective, under-samples the fully-sampled data with generated mask.'
                                       'If prospective, runs score-POCS with the given mask')
    parser.add_argument('--data', type=str, help='which data to use for reconstruction', required=True)
    parser.add_argument('--mask_type', type=str, help='which mask to use for retrospective undersampling.'
                                                      '(NOTE) only used for retrospective model!', default='gaussian1d',
                        choices=['gaussian1d', 'uniform1d', 'gaussian2d'])
    parser.add_argument('--acc_factor', type=int, help='Acceleration factor for Fourier undersampling.'
                                                       '(NOTE) only used for retrospective model!', default=4)
    parser.add_argument('--center_fraction', type=float, help='Fraction of ACS region to keep.'
                                                       '(NOTE) only used for retrospective model!', default=0.08)
    parser.add_argument('--save_dir', default='./results')
    parser.add_argument('--N', type=int, help='Number of iterations for score-POCS sampling', default=2000)
    parser.add_argument('--m', type=int, help='Number of corrector step per single predictor step.'
                                              'It is advised not to change this default value.', default=1)
    return parser


if __name__ == "__main__":
    main()
