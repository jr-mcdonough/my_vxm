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
parser.add_argument('--reg_model_dir', required=True, help='directory to load registration model checkpoints')
parser.add_argument('--atlas_model_dir', required=True, help='directory to load atlas model checkpoints')
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
atlas_path = args.atlas_model_dir
reg_path = args.reg_model_dir

# get all .pt model checkpoints in the folder
atlas_model_paths = glob.glob(os.path.join(atlas_path, '*.pt'))
reg_model_paths = glob.glob(os.path.join(reg_path, '*.pt'))

# sort the lists just in case
atlas_model_paths.sort()
reg_model_paths.sort()

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

segsample1s_moved = []

# convert the sample image data for model evaluation
insample = [torch.from_numpy(d).to(device).float().permute(0, 3, 1, 2) for d in insample]
outsample = [torch.from_numpy(d).to(device).float().permute(0, 3, 1, 2) for d in outsample]
segsample = [torch.from_numpy(d).to(device).float().permute(0, 3, 1, 2) for d in segsample]

for atlas_checkpoint, reg_checkpoint in zip(atlas_model_paths, reg_model_paths):

    atlas_model = vxm.myutils.VxmAtlas.load(path=atlas_checkpoint, device=device)

    reg_model = vxm.networks.VxmDense(
            inshape=(160, 160),
            nb_unet_features=[[16, 32, 32, 32], [32, 32, 32, 32, 32, 16, 16]],
            bidir=True,
            int_steps=7,
            int_downsize=2
        )

    reg_model.to(device)
    reg_model.load(path=reg_checkpoint, device=device)

    atlas_model.eval()
    reg_model.eval()

    # initialize lists to evaluate model performance
    model_loss = []
    model_total_loss = []
    dice_score = []
    dice_score_no_bg = []


    # get the current atlas
    atlas = atlas_model().to(device)
    
    for _ in range(100):
        
        # generator returns [scan1, scan2], [scan2, scan1, zeros], [seg1, seg2]
        moving, fixed, moving_segs = next(generator)

        scan1 = torch.from_numpy(moving[0]).to(device).float().permute(0, 3, 1, 2)
        scan2 = torch.from_numpy(moving[1]).to(device).float().permute(0, 3, 1, 2)

        # break up scan1, scan2 to register through atlas
        img1_atl = [scan1, atlas]
        atl_img2 = [atlas, scan2]

        fixed = [torch.from_numpy(d).to(device).float().permute(0, 3, 1, 2) for d in fixed]
        moving_segs = [torch.from_numpy(d).to(device).float().permute(0, 3, 1, 2) for d in moving_segs]

        # get the ground truth outputs for registering through atlas
        y_true1 = [atlas, scan1, fixed[-1]]
        y_true2 = [scan2, atlas, fixed[-1]]

        seg1 = moving_segs[0]
        seg2 = moving_segs[1]

        # reg_model returns[atlas_pred, scan1_pred, flow_1toatl] for first stage of registration
        pred_atl_img1 = reg_model(*img1_atl)

        # measure the loss for first stage of registration
        loss = 0
        loss_list = []
        for n, loss_function in enumerate(losses):
            curr_loss = loss_function(y_true1[n], pred_atl_img1[n]) * weights[n]
            loss_list.append(curr_loss.item())
            loss += curr_loss

        model_loss.append(loss_list)
        model_total_loss.append(loss.item())

        # separate the moved images and velocity field
        moved = pred_atl_img1[:-1] # [atlas_pred, scan1_pred]
        flow = pred_atl_img1[-1] # flow_1toatl

        # put the segmentations through the integrated flow field
        moved_segs = reg_model.move_image(flow, seg1, seg2)
        seg1_atl = moved_segs[0]

        # reg_model returns[scan2_pred, atl_pred, flow_atlto2] for second stage of registration
        pred_img2_atl = reg_model(*atl_img2)

        # measure the loss for second stage of registration
        loss = 0
        loss_list = []
        for n, loss_function in enumerate(losses):
            curr_loss = loss_function(y_true2[n], pred_img2_atl[n]) * weights[n]
            loss_list.append(curr_loss.item())
            loss += curr_loss

        model_loss.append(loss_list)
        model_total_loss.append(loss.item())

        # separate the moved images and velocity field
        moved = pred_img2_atl[:-1] # [scan2_pred, atl_pred]
        flow = pred_img2_atl[-1] # flow_atlto2

        # put the segmentations through the integrated flow field
        # take output of first stage of registration, ground truth for second stage
        # flow field of second stage
        moved_segs = reg_model.move_image(flow, seg1_atl, seg2)
        seg2_pred = moved_segs[0]

        # measure the Dice score between moving, moved images
        dice_with_bg, dice_without_bg = dice_eval.score(seg2, seg2_pred, num_classes=25)

        dice_score.append(dice_with_bg.item())
        dice_score_no_bg.append(dice_without_bg.item())

    
    # for the sample images:

    # arrange the sample images and segmentations to register through the atlas
    img1_atl = [insample[0], atlas]
    atl_img2 = [atlas, insample[1]]


    # register first sample to atlas to get first stage of flow field
    pred_atl_img1 = reg_model(*img1_atl)

    sample_moved1 = pred_atl_img1[:-1]
    sample_flow1 = pred_atl_img1[-1]

    # register their segmentations
    segsample_moved1 = reg_model.move_image(sample_flow1, *segsample)

    # register atlas to second sample to get second stage of flow field
    pred_img2_atl = reg_model(*atl_img2)

    sample_moved2 = pred_atl_img1[:-1]
    sample_flow2 = pred_atl_img1[-1]

    # register predicted atlas and segmentation from scan1 with this flow field to get scan2 predictions
    sample_final = reg_model.move_image(sample_flow2, *sample_moved1)
    segsample_final = reg_model.move_image(sample_flow2, *segsample_moved1)
    
    # record the moved samples, segmentations, and flow field for the model

    # reformat the data from torch to numpy
    sample_final = [d.permute(0, 2, 3, 1).cpu().detach().squeeze().numpy() for d in sample_final]
    segsample_final = [d.permute(0, 2, 3, 1).cpu().detach().squeeze().numpy() for d in segsample_final]

    # append data to lists
    sample1s_moved.append(sample_final[0])
    segsample1s_moved.append(segsample_final[0])


    # print the model performance data
    model_info = reg_checkpoint + '\n'
    losses_info = ', '.join(['%.4e' % f for f in np.mean(model_loss, axis=0)])
    loss_info = 'loss: %.4e  (%s)' % (np.mean(model_total_loss), losses_info)
    val_info = 'mean dice score: %.4e, mean dice score (no background) %.4e' % (np.mean(dice_score), np.mean(dice_score_no_bg))
    print(' - '.join((model_info, loss_info, val_info)), flush=True)

    # record the average loss, Dice score for each checkpoint
    loss_data.append(np.mean(model_total_loss))
    dice_data.append(np.mean(dice_score))
    dice_no_bg_data.append(np.mean(dice_score_no_bg))

    # delete the model and free up the cached memory on GPU
    del reg_model
    del atlas_model
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

# registration, segmentation data from each model
moved_samples_dict = {'sample1_moved': sample1s_moved,
                      'segsample1_moved': segsample1s_moved}

np.savez(os.path.join(args.eval_dir, 'moved_samples_dict'),
         sample1_moved=[arr for arr in moved_samples_dict["sample1_moved"]],
         segsample1_moved=[arr for arr in moved_samples_dict["segsample1_moved"]])
