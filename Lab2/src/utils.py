import torch
import torch.nn


def dice_score(pred_mask, gt_mask):
    # Input Size of pred_mask and gt_mask : (Batch, 1, H, W)

    # print(pred_mask.shape)
    # print(gt_mask.shape)

    # trimaps/ 	Trimap annotations for every image in the dataset
    # Pixel Annotations: 1: Foreground 2:Background 3: Not classified
    # We should calculate  the dice score based on label 1 (pixel 1 and pixel 3 are labled as 1)

    smooth = 1e-6  # add smooth value in case of 0 value denominator and nominator

    pred_mask = (pred_mask > 0.5).float()
    # We only flatten each sample in a batch
    pred_mask = torch.flatten(pred_mask, start_dim=1)
    gt_mask = torch.flatten(gt_mask, start_dim=1)

    # Calculate the intersection of label 1 and sum up the number of label 1 of each sample
    # multiply(*) sign indicates that operates element-wise multiplication
    intersection = (pred_mask * gt_mask).sum(dim=1)
    union = pred_mask.sum(dim=1) + gt_mask.sum(dim=1)

    dice = (2.0 * intersection + smooth) / (union + smooth)

    return dice.mean()


def dice_loss(pred_mask, gt_mask):
    smooth = 1e-6

    # We do not perform Binarization here，since this will cause dice_lose nondifferentiable

    # We only flatten each sample in a batch
    pred_mask = torch.flatten(pred_mask, start_dim=1)
    gt_mask = torch.flatten(gt_mask, start_dim=1)

    # Calculate the intersection of label 1 and sum up the number of label 1 of each sample
    # multiply(*) sign indicates that operates element-wise multiplication
    intersection = (pred_mask * gt_mask).sum(dim=1)
    union = pred_mask.sum(dim=1) + gt_mask.sum(dim=1)

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()
