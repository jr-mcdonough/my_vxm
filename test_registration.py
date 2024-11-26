import os, sys
import glob
import numpy as np
import random
import time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('/content/drive/My Drive/EC500')

import vxm_torch as vxm

# parse command line arguments
parser = argparse.ArgumentParser()

# data preparation parameters
parser.add_argument('--imgs_source', required=True, help='source of image data: dir or dict')
parser.add_argument('--imgs_path', required=True, help='path to image data')
parser.add_argument('--model_dir', required=True, help='directory to load model checkpoints')
parser.add_argument('--eval_dir', required=True, help='directory to save model evaluation data')

args = parser.parse_args()

# load and format the test images and segmentations
imgs = []
segs = []

# if a directory to images is provided, read nii.gz files
if args.imgs_source == 'dir':

    imgs_dir = args.imgs_path

    for item in os.listdir(imgs_dir):
        item_path = os.path.join(imgs_dir, item)

        if os.path.isdir(item_path):
            read_img, read_seg = vxm.myutils.load_nii_gz(item_path, load_seg=True)
            imgs.append(read_img)
            segs.append(read_seg)

    assert len(imgs) == len(segs), 'Number of images and segmentations are not equal'
    print(f'{len(imgs)} training images and segmentations loaded')

# if a dictionary of images is provided, read .npz file
elif args.imgs_source == 'dict':

    imgs_dict = np.load(args.imgs_path, allow_pickle=True)

    imgs = list(imgs_dict["imgs"])
    segs = list(imgs_dict['segs'])

else:
    print('--img_source must be dir or dict')
    sys.exit()

# put the test data into a generator
generator = vxm.generators.scan_to_scan(vol_names=imgs, bidir=True, batch_size=1, segs=segs)

# draw two sample images to compare across different models
insample, outsample, segsample = next(generator)


# load the paths to the model checkpoints
folder_path = args.model_dir

# get all .pt model checkpoints in the folder
model_paths = glob.glob(os.path.join(folder_path, '*.pt'))

# sort the list just in case
model_paths.sort()

# define losses, Dice score for evaluating model
losses = [vxm.losses.MSE().loss, vxm.losses.MSE().loss, vxm.losses.Grad('l2', loss_mult=2).loss]
weights = [0.5, 0.5, 0.01]

dice_eval = vxm.myutils.MulticlassDiceScore()


# run the test data through each model
device = 'cuda'

# prepare lists to store test loss, test Dice scores
loss_data=[]
dice_data=[]
dice_no_bg_data=[]

# prepare lists to store sample pair data
sample1s_moved = []
sample2s_moved = []

segsample1s_moved = []
segsample2s_moved = []

sample_flows = []

# convert the sample image data for model evaluation
insample = [torch.from_numpy(d).to(device).float().permute(0, 3, 1, 2) for d in insample]
outsample = [torch.from_numpy(d).to(device).float().permute(0, 3, 1, 2) for d in outsample]
segsample = [torch.from_numpy(d).to(device).float().permute(0, 3, 1, 2) for d in segsample]

for checkpoint in model_paths:

    model = vxm.networks.VxmDense(
            inshape=(160, 160),
            nb_unet_features=[[16, 32, 32, 32], [32, 32, 32, 32, 32, 16, 16]],
            bidir=True,
            int_steps=7,
            int_downsize=2
        )

    model.to(device)
    model.load(path=checkpoint, device=device)
    model.eval()

    # initialize lists to evaluate model performance
    model_loss = []
    model_total_loss = []
    dice_score = []
    dice_score_no_bg = []


    for _ in range(100):
        
        inputs, y_true, segs = next(generator)

        inputs = [torch.from_numpy(d).to(device).float().permute(0, 3, 1, 2) for d in inputs]
        y_true = [torch.from_numpy(d).to(device).float().permute(0, 3, 1, 2) for d in y_true]
        segs = [torch.from_numpy(d).to(device).float().permute(0, 3, 1, 2) for d in segs]

        y_pred = model(*inputs)

        loss = 0
        loss_list = []
        for n, loss_function in enumerate(losses):
            curr_loss = loss_function(y_true[n], y_pred[n]) * weights[n]
            loss_list.append(curr_loss.item())
            loss += curr_loss

        model_loss.append(loss_list)
        model_total_loss.append(loss.item())

        flow = y_pred[-1]

        moved_segs = model.move_image(flow, *segs)

        for mvg, mvd in zip(segs, moved_segs):

            dice_with_bg, dice_without_bg = dice_eval.score(mvg, mvd, num_classes=25)
            
            dice_score.append(dice_with_bg.item())
            dice_score_no_bg.append(dice_without_bg.item())

    
    # for the sample images:

    # register them
    sample_pred = model(*insample)

    sample_moved = sample_pred[:-1]
    sample_flow = sample_pred[-1]

    # register their segmentations
    segsample_moved = model.move_image(sample_flow, *segsample)

    # record the moved samples, segmentations, and flow field for the model

    # reformat the data from torch to numpy
    sample_moved = [d.permute(0, 2, 3, 1).cpu().detach().squeeze().numpy() for d in sample_moved]
    sample_flow = sample_flow.permute(0, 2, 3, 1).cpu().detach().squeeze().numpy()
    segsample_moved = [d.permute(0, 2, 3, 1).cpu().detach().squeeze().numpy() for d in segsample_moved]

    # append data to lists
    sample1s_moved.append(sample_moved[0])
    sample2s_moved.append(sample_moved[1])

    segsample1s_moved.append(segsample_moved[0])
    segsample2s_moved.append(segsample_moved[1])

    sample_flows.append(sample_flow)


    # print the model performance data
    model_info = checkpoint + '\n'
    losses_info = ', '.join(['%.4e' % f for f in np.mean(model_loss, axis=0)])
    loss_info = 'loss: %.4e  (%s)' % (np.mean(model_total_loss), losses_info)
    val_info = 'mean dice score: %.4e, mean dice score (no background) %.4e' % (np.mean(dice_score), np.mean(dice_score_no_bg))
    print(' - '.join((model_info, loss_info, val_info)), flush=True)

    # record the average loss, Dice score for each checkpoint
    loss_data.append(np.mean(model_total_loss))
    dice_data.append(np.mean(dice_score))
    dice_no_bg_data.append(np.mean(dice_score_no_bg))

    # delete the model and free up the cached memory on GPU
    del model
    torch.cuda.empty_cache()

# save the test loss and Dice scores for each model
loss_data = np.array(loss_data)
np.save(os.path.join(args.eval_dir, 'loss_data'), loss_data)

dice_data = np.array(dice_data)
np.save(os.path.join(args.eval_dir, 'dice_data'), dice_data)

dice_no_bg_data = np.array(dice_no_bg_data)
np.save(os.path.join(args.eval_dir, 'dice_no_bg_data'), dice_no_bg_data)

# save the data from sample pair of images

# ground truth data
sample1 = insample[0].permute(0, 2, 3, 1).cpu().detach().squeeze().numpy()
sample2 = insample[1].permute(0, 2, 3, 1).cpu().detach().squeeze().numpy()

segsample1 = segsample[0].permute(0, 2, 3, 1).cpu().detach().squeeze().numpy()
segsample2 = segsample[1].permute(0, 2, 3, 1).cpu().detach().squeeze().numpy()

sample_dict = {'sample1': sample1, 'sample2': sample2, 'segsample1': segsample1, 'segsample2': segsample2}
np.savez(os.path.join(args.eval_dir, 'sample_dict'), **sample_dict)

# registration, segmentation, flow data from each model
moved_samples_dict = {'sample1_moved': sample1s_moved,
                      'sample2_moved': sample2s_moved,
                      'segsample1_moved': segsample1s_moved,
                      'segsample2_moved': segsample2s_moved}

np.savez(os.path.join(args.eval_dir, 'moved_samples_dict'),
         sample1_moved=[arr for arr in moved_samples_dict["sample1_moved"]],
         sample2_moved=[arr for arr in moved_samples_dict["sample2_moved"]],
         segsample1_moved=[arr for arr in moved_samples_dict["segsample1_moved"]],
         segsample2_moved=[arr for arr in moved_samples_dict["segsample2_moved"]])


flows_dict = {'flows': sample_flows}

np.savez(os.path.join(args.eval_dir, 'flows_dict'),
         flow=[arr for arr in flows_dict["flows"]])
