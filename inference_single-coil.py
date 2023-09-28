from pathlib import Path
from models import utils as mutils
from sde_lib import VESDE
from sampling import (ReverseDiffusionPredictor,
                      LangevinCorrector,
                      get_pc_fouriercs_RI)
import os
import ntpath
from models import ncsnpp
import time
from utils import fft2, ifft2, fft2_m, ifft2_m, get_mask, get_data_scaler, get_data_inverse_scaler, restore_checkpoint
import torch
import torch.nn as nn
import numpy as np
from models.ema import ExponentialMovingAverage
import matplotlib.pyplot as plt
import importlib
import argparse
from typing import List, Optional
import argparse
import h5py
import torchvision.transforms as T
import torchvision
import imageio

def normalize_complex_arr(a):
    a_oo = a - a.real.min() - 1j*a.imag.min()
    return a_oo/np.abs(a_oo).max()


def main():
    ###############################################
    # 1. Configurations
    ###############################################

    # args
    args = create_argparser().parse_args()
    N = args.N
    m = args.m
    fpath = args.data
    head_path, fname_ = os.path.split(os.path.join(fpath))
    print(head_path)
    print(fname_)
    #filename = f'./samples/single-coil/{fname}.npy'

    print('initializing...')
    configs = importlib.import_module(f"configs.ve.fastmri_knee_320_ncsnpp_continuous")
    config = configs.get_config()
    img_size = config.data.image_size
    batch_size = 1

    # Read data
    #img = torch.from_numpy(np.load(filename).astype(np.complex64))
    #print(np.median(img.real))

    #file = "/vol/datasets/cil/2021_11_23_fastMRI_data/knee/unzipped/singlecoil_train/file1000898.h5"
    #file = "/vol/datasets/cil/2021_11_23_fastMRI_data/knee/unzipped/singlecoil_test_v2/file1000692_v2.h5"

    file = fpath
    hf = h5py.File(file, 'r')
    print(hf.keys())
    print(hf['kspace'].shape)
    #k=hf['kspace'][hf['kspace'].shape[0]//2,:,:]

    for i in range(hf['kspace'].shape[0]):
        k=hf['kspace'][i,:,:]
        fname = str(i) + "_" + fname_

        kt = torch.from_numpy(np.stack((k.real, k.imag), axis=-1))
        print(kt.dtype)
        ifftimg = ifft2_m(torch.from_numpy(k))#torch.fft.ifft2(torch.fft.ifftshift(kt))#.unsqueeze(0)#
        print(ifftimg.shape)

        cropx = (ifftimg.shape[0]-320)//2
        cropy = (ifftimg.shape[1]-320)//2 
        ifftimg=ifftimg[cropx:cropx+320,cropy:cropy+320]
        print(ifftimg.shape)

        #TODO use kspace directly and convert back with the functions here
        img = normalize_complex_arr(ifftimg)
        print(np.median(img.real))

        img = img.view(1, 1, 320, 320)
        img = img.to(config.device)

        mask = get_mask(img, img_size, batch_size,
                        type=args.mask_type,
                        acc_factor=args.acc_factor,
                        center_fraction=args.center_fraction)
        
        print(mask.shape)
        mask = torch.ones(mask.size()).to(config.device)

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
        save_root = Path(f'./results/single-coil')
        save_root.mkdir(parents=True, exist_ok=True)

        irl_types = ['input', 'recon', 'recon_progress', 'label']
        for t in irl_types:
            save_root_f = save_root / t
            save_root_f.mkdir(parents=True, exist_ok=True)

        ###############################################
        # 2. Inference
        ###############################################

        pc_fouriercs = get_pc_fouriercs_RI(sde,
                                        predictor, corrector,
                                        inverse_scaler,
                                        snr=snr,
                                        n_steps=m,
                                        probability_flow=probability_flow,
                                        continuous=config.training.continuous,
                                        denoise=True)
        # fft
        kspace = fft2(img)
        plt.clf()
        plt.imshow(torch.log(torch.abs(kspace.cpu().squeeze(0).squeeze(0).real)).numpy(), cmap='gray')
        plt.savefig('sc_scfftinput_crop.png')

        # undersampling
        under_kspace = kspace * mask
        under_img = ifft2(under_kspace)
        plt.clf()
        plt.imshow(torch.abs(under_img.cpu().squeeze(0).squeeze(0).real).numpy(), cmap='gray')
        plt.savefig('sc_under_img_crop.png')

        print(f'Beginning inference')
        tic = time.time()
        x = pc_fouriercs(score_model, scaler(under_img), mask, Fy=under_kspace)
        toc = time.time() - tic
        print(f'Time took for recon: {toc} secs.')

        ###############################################
        # 3. Saving recon
        ###############################################
        input = under_img.squeeze().cpu().detach().numpy()
        label = img.squeeze().cpu().detach().numpy()
        mask_sv = mask.squeeze().cpu().detach().numpy()
        under_kspace_sv = under_kspace.squeeze().cpu().detach().numpy()
        kspace_sv = kspace.squeeze().cpu().detach().numpy()

        np.save(str(save_root / 'input' / fname) + '.npy', input)
        np.save(str(save_root / 'input' / (fname + '_mask')) + '.npy', mask_sv)
        np.save(str(save_root / 'label' / fname) + '.npy', label)
        plt.imsave(str(save_root / 'input' / fname) + '_kspace.png', np.log(np.abs(kspace_sv)), cmap='gray')
        plt.imsave(str(save_root / 'input' / fname) + '_under_kspace.png', np.log(np.abs(under_kspace_sv)), cmap='gray')
        plt.imsave(str(save_root / 'input' / fname) + '.png', np.abs(input), cmap='gray')
        plt.imsave(str(save_root / 'label' / fname) + '.png', np.abs(label), cmap='gray')
        plt.imsave(str(save_root / 'input' / fname) + '_mask.png', np.abs(mask_sv), cmap='gray')

        recon = x.squeeze().cpu().detach().numpy()
        np.save(str(save_root / 'recon' / fname) + '.npy', recon)
        plt.imsave(str(save_root / 'recon' / fname) + '.png', np.abs(recon), cmap='gray')


def create_argparser():
    parser = argparse.ArgumentParser()
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