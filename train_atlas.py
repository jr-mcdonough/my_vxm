import os, sys
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
parser.add_argument('--val_size', default=None, help='size of validation set')
parser.add_argument('--pretrained_reg_path', default=None, help='path to pretrained registration model weights')
parser.add_argument('--reg_model_dir', required=True, help='directory to save registration model checkpoints')
parser.add_argument('--atlas_model_dir', required=True, help='directory to save atlas model checkpoints')
parser.add_argument('--loss_dir', required=True, help='directory to save epoch loss data')
parser.add_argument('--num_epochs', default=1500, help='number of training epochs')

# training parameters

# network parameters

# training hyper-parameters

args = parser.parse_args()


# load and format the training (and validation) images and segmentations
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


# prepare a validation set if one is specified
val_size=args.val_size

if val_size is not None:

    indices = list(range(len(imgs)))
    
    val_indices = random.sample(indices, val_size)
    train_indices = list(set(indices) - set(val_indices))

    train_imgs = [imgs[i] for i in train_indices]

    val_imgs = [imgs[i] for i in val_indices]
    val_segs = [segs[i] for i in val_indices]

else:

    train_imgs = imgs

# make an initial atlas by averaging 100 training images
atlas_samples = random.sample(train_imgs, 100)
initial_atlas = np.mean(atlas_samples, axis=0)

# reshape it to be compatible with the format of training images later
initial_atlas = initial_atlas.reshape((1, *initial_atlas.shape, 1))


# prepare the necessary data generators
if val_size is not None:
    train_generator = vxm.generators.scan_to_scan(vol_names=train_imgs, bidir=True, batch_size=1, atlas=initial_atlas, segs=None)
    val_generator = vxm.generators.scan_to_scan(vol_names=val_imgs, bidir=True, batch_size=1, segs=val_segs)

else:
    train_generator = vxm.generators.scan_to_scan(vol_names=train_imgs, bidir=True, batch_size=1, atlas=initial_atlas, segs=None)

# make an atlas model with this initial atlas
device='cuda'

atlas_model = vxm.myutils.VxmAtlas(initial_atlas)
atlas_model.to(device)
atlas_model.train()

# instantiate a registration model

reg_model = vxm.networks.VxmDense(
        inshape=(160, 160),
        nb_unet_features=[[16, 32, 32, 32], [32, 32, 32, 32, 32, 16, 16]],
        bidir=True,
        int_steps=7,
        int_downsize=2
    )

reg_model.to(device)

# optionally load a pre-trained registration weights

if args.pretrained_reg_path is not None:
    reg_model.load(path=args.pretrained_reg_path, device=device)

    # freeze all layers in the Unet
    for param in reg_model.unet_model.parameters():
        param.requires_grad = False

    # unfreeze the last layers of the Unet
    # convolutions at full resolution
    for param in reg_model.unet_model.remaining.parameters():
        param.requires_grad = True

    # convolution to reshape to flow field
    for param in reg_model.flow.parameters():
        param.requires_grad = True

    # verify which parameters are being updated
    for name, param in reg_model.named_parameters():
        if param.requires_grad:
            print(f"Trainable layer: {name}")

reg_model.train()

# set the optimizer and losses for the network

if args.pretrained_reg_path is not None:
    reg_optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, reg_model.parameters()), 
        lr=1e-4
    )

else:
    reg_optimizer = torch.optim.Adam(reg_model.parameters(), lr=1e-4)

atlas_optimizer = torch.optim.Adam(atlas_model.parameters(), lr=1e-4)

losses = [vxm.losses.MSE().loss, vxm.losses.MSE().loss, vxm.losses.Grad('l2', loss_mult=2).loss]
weights = [0.5, 0.5, 0.01]

dice_eval = vxm.myutils.MulticlassDiceScore()


# training loop

# save data to plot training loss
loss_data = []

for epoch in range(args.num_epochs):

    # save model checkpoint
    if epoch % 20 == 0:
        atlas_model.save(os.path.join(args.atlas_model_dir, 'atlas_%04d.pt' % epoch))
        reg_model.save(os.path.join(args.reg_model_dir, '%04d.pt' % epoch))

    epoch_loss = []
    epoch_total_loss = []
    epoch_step_time = []

    for step in range(100):

        step_start_time = time.time()

        # get the current atlas
        atlas = atlas_model()
        
        # generate inputs (and true outputs) and convert them to tensors
        # generator returns [scan1, initial_atlas], [initial_atlas, scan1, zeros]
        inputs, y_true = next(train_generator)

        scan1 = torch.from_numpy(inputs[0]).to(device).float().permute(0, 3, 1, 2)
        zeros = y_true[-1]

        inputs = [atlas, scan1]
        y_true = [scan1, atlas, zeros]

        # run inputs through the model to produce a warped image and flow field
        # reg_model returns (pred_scan1, pred_atlas, flow)
        y_pred = reg_model(*inputs)

        # calculate total loss
        loss = 0
        loss_list = []
        for n, loss_function in enumerate(losses):
            curr_loss = loss_function(y_true[n], y_pred[n]) * weights[n]
            loss_list.append(curr_loss.item())
            loss += curr_loss

        epoch_loss.append(loss_list)
        epoch_total_loss.append(loss.item())

        # backpropagate and optimize
        atlas_optimizer.zero_grad()
        reg_optimizer.zero_grad()
        loss.backward()
        atlas_optimizer.step()
        reg_optimizer.step()

        # get compute time
        epoch_step_time.append(time.time() - step_start_time)

    # validation step
    # register a scan1 to scan2 through the current atlas
    if val_size is not None:
        val_dice = []
        val_dice_no_bg = []

        with torch.no_grad():

            # generate inputs, outputs, and segmentations and convert them to tensors
            for step in range(val_size):

                # val generator returns [scan1, scan2], [scan2, scan1, zeros], [seg1, seg2]
                moving, fixed, moving_segs = next(val_generator)

                scan1 = torch.from_numpy(moving[0]).to(device).float().permute(0, 3, 1, 2)
                scan2 = torch.from_numpy(moving[1]).to(device).float().permute(0, 3, 1, 2)

                img1_atl = [scan1, atlas]
                atl_img2 = [atlas, scan2]

                moving_segs = [torch.from_numpy(d).to(device).float().permute(0, 3, 1, 2) for d in moving_segs]
                seg1 = moving_segs[0]
                seg2 = moving_segs[1]

                # reg_model returns[atlas_pred, scan1_pred, flow_1to2] for first stage of registration
                val_pred = reg_model(*img1_atl)

                # separate the moved images and velocity field
                moved = val_pred[:-1] # [atlas_pred, scan1_pred]
                flow = val_pred[-1] # flow_1toatl

                # put the segmentations through the integrated flow field
                moved_segs = reg_model.move_image(flow, seg1, seg2)
                seg1_atl = moved_segs[0]

                # reg_model returns[scan2_pred, atl_pred, flow_atlto2] for second stage of registration
                val_pred = reg_model(*atl_img2)

                # separate the moved images and velocity field
                moved = val_pred[:-1] # [scan2_pred, atl_pred]
                flow = val_pred[-1] # flow_atlto2

                # put the segmentations through the integrated flow field
                # take output of first stage of registration, ground truth for second stage
                # flow field of second stage
                moved_segs = reg_model.move_image(flow, seg1_atl, seg2)
                seg2_pred = moved_segs[0]

                # measure the Dice score between moving, moved images
                dice_with_bg, dice_without_bg = dice_eval.score(seg2, seg2_pred, num_classes=25)

                val_dice.append(dice_with_bg.item())
                val_dice_no_bg.append(dice_without_bg.item())

    # print epoch info
    epoch_info = 'Epoch %d/%d' % (epoch + 1, args.num_epochs)
    time_info = '%.4f sec/step' % np.mean(epoch_step_time)
    losses_info = ', '.join(['%.4e' % f for f in np.mean(epoch_loss, axis=0)])
    loss_info = 'loss: %.4e  (%s)' % (np.mean(epoch_total_loss), losses_info)

    if val_size is not None:
        val_info = 'validation dice: %.4e, validation dice (no background) %.4e' % (np.mean(val_dice), np.mean(val_dice_no_bg))
        print(' - '.join((epoch_info, time_info, loss_info, val_info)), flush=True)

    else:
        print(' - '.join((epoch_info, time_info, loss_info)), flush=True)

    loss_data.append(np.mean(epoch_total_loss))

# final model save
atlas_model.save(os.path.join(args.atlas_model_dir, 'atlas_%04d.pt' % args.num_epochs))
reg_model.save(os.path.join(args.reg_model_dir, '%04d.pt' % args.num_epochs))

# save training loss data
loss_data = np.array(loss_data)
np.save(os.path.join(args.loss_dir, 'loss_data'), loss_data)
