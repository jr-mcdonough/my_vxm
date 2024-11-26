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
parser.add_argument('--model_dir', required=True, help='directory to save model checkpoints')
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
# prepare the necessary data generators
val_size=args.val_size

if args.val_size is not None:

    indices = list(range(len(imgs)))
    
    val_indices = random.sample(indices, val_size)
    train_indices = list(set(indices) - set(val_indices))

    train_imgs = [imgs[i] for i in train_indices]

    val_imgs = [imgs[i] for i in val_indices]
    val_segs = [segs[i] for i in val_indices]

    train_generator = vxm.generators.scan_to_scan(vol_names=train_imgs, bidir=True, batch_size=1, segs=None)
    val_generator = vxm.generators.scan_to_scan(vol_names=val_imgs, bidir=True, batch_size=1, segs=val_segs)

else:

    train_imgs = imgs
    train_generator = vxm.generators.scan_to_scan(vol_names=train_imgs, bidir=True, batch_size=1, segs=None)


# instantiate a registration model
model = vxm.networks.VxmDense(
        inshape=(160, 160),
        nb_unet_features=[[16, 32, 32, 32], [32, 32, 32, 32, 32, 16, 16]],
        bidir=True,
        int_steps=7,
        int_downsize=2
    )

device='cuda'

model.to(device)
model.train()

# set the optimizer and losses for the network
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

losses = [vxm.losses.MSE().loss, vxm.losses.MSE().loss, vxm.losses.Grad('l2', loss_mult=2).loss]
weights = [0.5, 0.5, 0.01]

dice_eval = vxm.myutils.MulticlassDiceScore()


# training loop

# save data to plot training loss
loss_data = []

for epoch in range(args.num_epochs):

    # save model checkpoint
    if epoch % 20 == 0:
        model.save(os.path.join(args.model_dir, '%04d.pt' % epoch))

    epoch_loss = []
    epoch_total_loss = []
    epoch_step_time = []

    for step in range(100):

        step_start_time = time.time()

        # generate inputs (and true outputs) and convert them to tensors
        inputs, y_true = next(train_generator)
        inputs = [torch.from_numpy(d).to(device).float().permute(0, 3, 1, 2) for d in inputs] # changed .permute(0, 4, 1, 2, 3) -> .permute(0, 3, 1, 2)
        y_true = [torch.from_numpy(d).to(device).float().permute(0, 3, 1, 2) for d in y_true] # to acount for change from 3D -> 2D dataset

        # run inputs through the model to produce a warped image and flow field
        y_pred = model(*inputs)

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
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # get compute time
        epoch_step_time.append(time.time() - step_start_time)

    # validation step
    if val_size is not None:
        val_dice = []
        val_dice_no_bg = []

        with torch.no_grad():

            # generate inputs, outputs, and segmentations and convert them to tensors
            for step in range(val_size):

                moving, fixed, moving_segs = next(val_generator)

                moving = [torch.from_numpy(d).to(device).float().permute(0, 3, 1, 2) for d in moving]
                fixed = [torch.from_numpy(d).to(device).float().permute(0, 3, 1, 2) for d in fixed]
                moving_segs = [torch.from_numpy(d).to(device).float().permute(0, 3, 1, 2) for d in moving_segs]

                val_pred = model(*moving)

                # separate the moved images and velocity field
                moved = val_pred[:-1]
                flow = val_pred[-1]

                # put the segmentations through the integrated flow field
                moved_segs = model.move_image(flow, *moving_segs)

                # measure the Dice score between moving, moved images
                for mvg, mvd in zip(moving_segs, moved_segs):

                    dice_with_bg, dice_without_bg = dice_eval.score(mvg, mvd, num_classes=25)
                    
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
model.save(os.path.join(args.model_dir, '%04d.pt' % args.num_epochs))

# save training loss data
loss_data = np.array(loss_data)
np.save(os.path.join(args.loss_dir, 'loss_data'), loss_data)
