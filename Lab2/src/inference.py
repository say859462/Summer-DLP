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
from utils import dice_score


def inference(args, device, model):

    assert args.model in {"Unet", "ResNet34_Unet"}
    tqdm.write("Testing model {}".format(args.model))
    test_data = load_dataset(args.data_path, mode="test", transform=None)
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

            test_loss.append(criterion(pred_mask, mask).item())
            test_dice_scores.append(dice_score(pred_mask, mask).item())

            # ----------Store result image Begin----------
            pred_mask = pred_mask.squeeze(0).squeeze(0).cpu().numpy()

            binary_mask = (pred_mask > 0.5).astype(np.uint8)

            np_image = (image.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(
                np.uint8
            )
            pil_image = Image.fromarray(np_image).convert("RGBA")

            np_rgba = np.array(pil_image)

            # lower alpha value of foreground
            # foreground：alpha=120；background：alpha=255
            np_rgba[..., 3] = np.where(binary_mask == 1, 120, 255)
            pil_result = Image.fromarray(np_rgba)
            pil_result.save(f"outputs_imgs/{args.model}/{cnt}.png")

            cnt += 1
            # ----------Store result image End----------
        tqdm.write(
            "Test loss: {:.4f}, Test Dice Score: {:.4f}".format(
                np.mean(test_loss), np.mean(test_dice_scores)
            )
        )

    if not os.path.exists("saved_metrics"):
        os.mkdir("saved_metrics")
    # Store loss and dice_score array for analyzing purpose
    np.save(f"saved_metrics/{args.model}_train_loss.npy", np.array(test_loss))
    np.save(f"saved_metrics/{args.model}_train_dice.npy", np.array(test_dice_scores))


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
        default="Unet",
        help="The model for tranining, Unet/ResNet34_Unet",
    )
    parser.add_argument(
        "--epochs", "-e", type=int, default=400, help="number of epochs"
    )
    parser.add_argument("--batch_size", "-b", type=int, default=16, help="batch size")
    parser.add_argument(
        "--learning_rate", "-lr", type=float, default=1e-4, help="learning rate"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
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

    os.makedirs("outputs_imgs/ResNet34_Unet", exist_ok=True)
    os.makedirs("outputs_imgs/Unet", exist_ok=True)

    inference(args, device, model)
