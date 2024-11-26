import os
import numpy as np
import nibabel as nib

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modelio import LoadableModel, store_config_args

def load_nii_gz(read_dir, load_seg=False, ret_affine=False):
    '''
    reads a single .nii.gz file and returns an image array, as well as an optional
    segmentation array and affine transform array. Hardcoded to rotate and clip image
    and segmentation arrays, based on manual evaluation of the dataset.

    assumes that input directory contains slice_norm.nii.gz, slice_seg24.nii.gz
    files for a single subject from 2D OASIS dataset

    params:
        read_dir (str): directory containing files to be read
        load_seg (bool): set true to return segmentation array
        ret_affine (bool): set true to return affine transform array

    returns:
        img (numpy.ndarray): MRI image
        seg (numpy.ndarray): 24 class segmentation
        affine (numpy.ndarray): affine transform
    '''
    img_path = os.path.join(read_dir, 'slice_norm.nii.gz')

    assert os.path.exists(img_path), f'No image file exists in {read_dir}'
    data = nib.load(img_path)
    img = np.squeeze(data.dataobj)
    affine = data.affine

    img = img[:, 1:161].T

    if load_seg:
        seg_path = os.path.join(read_dir, 'slice_seg24.nii.gz')

        assert os.path.exists(seg_path), f'No segmentation file exists in {read_dir}'
        seg = nib.load(seg_path)
        seg = np.squeeze(seg.dataobj)

        seg = seg[:, 1:161].T

        return (img, seg, affine) if ret_affine else (img, seg)

    return (img, affine) if ret_affine else img
    
    
class MulticlassDiceScore:
    """
    Multi-class Dice score for evaluation, with and without background.
    """

    def score(self, y_true, y_pred, num_classes=25):
        # Convert y_true and y_pred to one-hot encoded tensors
        y_true_onehot = F.one_hot(y_true.long(), num_classes).permute(0, 4, 1, 2, 3).float()
        y_pred_onehot = F.one_hot(y_pred.long(), num_classes).permute(0, 4, 1, 2, 3).float()

        # Calculate Dice score for each class
        vol_axes = [2, 3, 4]

        top = 2 * (y_true_onehot * y_pred_onehot).sum(dim=vol_axes)
        bottom = torch.clamp((y_true_onehot + y_pred_onehot).sum(dim=vol_axes), min=1e-5)
        dice_per_class = top / bottom

        # Calculate Dice score with and without background
        dice_with_bg = torch.mean(dice_per_class)  # Includes all classes
        dice_without_bg = torch.mean(dice_per_class[:, 1:])  # Excludes background (class 0)

        return dice_with_bg, dice_without_bg
        
        
class AtlasLayer(nn.Module):
    def __init__(self, initial_atlas):
        super(AtlasLayer, self).__init__()
        # Convert the initial atlas to a PyTorch tensor and make it trainable
        # Here we assume initial_atlas is a numpy array or tensor with shape (H, W, C) or similar
        initial_atlas_tensor = torch.from_numpy(initial_atlas).float().permute(0, 3, 1, 2)
        
        # Register the atlas as a trainable parameter
        self.atlas = nn.Parameter(initial_atlas_tensor)
        
    def forward(self):
        # Simply return the atlas; no input needed since it's a direct parameter representation
        return self.atlas
        

class VxmAtlas(LoadableModel):
    
    @store_config_args
    def __init__(self, initial_atlas):
        super().__init__()
        self.atlas_layer = AtlasLayer(initial_atlas)

    def forward(self):
        x = self.atlas_layer()

        return x