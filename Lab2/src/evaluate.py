from utils import dice_score, dice_loss
import torch
from torch import nn
from tqdm import tqdm
import numpy as np


def evaluate(model, valid_dataloader, device):

    # data : Validation data loader

    val_loss = []
    val_dice_scores = []
    criterion = nn.BCELoss()
    model.eval()
    with torch.no_grad():

        for data in valid_dataloader:
            image = data["image"].to(device)
            mask = data["mask"].to(device)
            pred_mask = model(image)
            loss = 0.5 * criterion(pred_mask, mask) + 1.5 * dice_loss(pred_mask, mask)

            val_loss.append(loss.item())
            val_dice_scores.append(dice_score(pred_mask, mask).item())

        tqdm.write(
            "Validation loss: {:.4f}, Validation Dice Score: {:.4f}".format(
                np.mean(val_loss), np.mean(val_dice_scores)
            )
        )

    return np.mean(val_loss), np.mean(val_dice_scores)
