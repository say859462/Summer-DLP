import argparse
import os

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.resnet34_unet import ResNet34_UNet
from models.unet import Unet
from oxford_pet import load_dataset
from train import train
from utils import dice_score, dice_loss, show_maskAndPredMask, show_maskOnimage


def inference(args, device, model):

    assert args.model in {"Unet", "ResNet34_Unet"}
    tqdm.write("Testing model {}".format(args.model))
    test_data = load_dataset(args.data_path, mode="test")
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)

    test_loss = []
    test_dice_scores = []

    criterion = nn.BCELoss()
    model.eval()
    cnt = 1
    with torch.no_grad():
        for data in tqdm(test_dataloader):

            image = data["image"].to(device)
            mask = data["mask"].to(device)

            pred_mask = model(image)
            loss = 0.5 * criterion(pred_mask, mask) + 1.5 * dice_loss(pred_mask, mask)
            test_loss.append(loss.item())

            test_dice_scores.append(dice_score(pred_mask, mask).item())

            # ----------Store result image Begin----------
            # show_maskOnimage(pred_mask, image, cnt, args)
            # show_maskAndPredMask(pred_mask, mask, cnt, args)
            cnt += 1
            # ----------Store result image End----------
        tqdm.write(
            "Test loss: {:.4f}, Test Dice Score: {:.4f}".format(
                np.mean(test_loss), np.mean(test_dice_scores)
            )
        )

    # if not os.path.exists("saved_metrics"):
    #     os.mkdir("saved_metrics")
    # # Store loss and dice_score array for analyzing purpose
    # np.save(f"saved_metrics/{args.model}_test_loss.npy", np.array(test_loss))
    # np.save(f"saved_metrics/{args.model}_test_dice.npy", np.array(test_dice_scores))


def get_args():
    parser = argparse.ArgumentParser(description="Predict masks from input images")

    parser.add_argument(
        "--data_path",
        type=str,
        default="dataset\oxford-iiit-pet",
        help="path of the input data",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="ResNet34_Unet",
        help="The model for tranining, Unet/ResNet34_Unet",
    )
    parser.add_argument(
        "--epochs", "-e", type=int, default=500, help="number of epochs"
    )
    parser.add_argument("--batch_size", "-b", type=int, default=16, help="batch size")
    parser.add_argument(
        "--learning_rate", "-lr", type=float, default=1e-4, help="learning rate"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    # Setting random seed for reproducibility
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = None

    # Load model if tranined , otherwise ,train it first
    if args.model == "Unet":
        model = Unet(3, 1).to(device)
        if os.path.exists("saved_models/Unet.pth"):
            model.load_state_dict(
                torch.load("saved_models/Unet.pth", weights_only=True)
            )
        else:
            train(args, device, model)
    else:
        model = ResNet34_UNet(3, 1).to(device)
        if os.path.exists("saved_models/ResNet34_Unet.pth"):
            model.load_state_dict(
                torch.load("saved_models/ResNet34_Unet.pth", weights_only=True)
            )
        else:
            train(args, device, model)

    # os.makedirs("outputs_imgs/stack/ResNet34_Unet", exist_ok=True)
    # os.makedirs("outputs_imgs/stack/Unet", exist_ok=True)
    # os.makedirs("outputs_imgs/non_stack/ResNet34_Unet", exist_ok=True)
    # os.makedirs("outputs_imgs/non_stack/Unet", exist_ok=True)
    # os.makedirs("outputs_imgs/combined/Unet", exist_ok=True)
    # os.makedirs("outputs_imgs/combined/ResNet34_Unet", exist_ok=True)

    inference(args, device, model)
