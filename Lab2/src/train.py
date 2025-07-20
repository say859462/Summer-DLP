import argparse
import os

import albumentations as A
import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import (
    ReduceLROnPlateau,
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
    StepLR,
)

from torch.utils.data import DataLoader
from tqdm import tqdm

import cv2
from evaluate import evaluate
from models.resnet34_unet import ResNet34_UNet
from models.unet import Unet
from oxford_pet import load_dataset
from utils import dice_score, dice_loss, show_result


def train(args, device, model):
    tqdm.write(
        "Training model {}, epoch {}, batch_size {}, learning_rate {}".format(
            args.model, args.epochs, args.batch_size, args.learning_rate
        )
    )
    assert args.model in {"Unet", "ResNet34_Unet"}
    lr_history = []
    train_loss = []
    train_dice_score = []
    val_loss = []
    val_dice_score = []
    best_dice_score = 0.0

    train_data = load_dataset(args.data_path, "train")
    valid_data = load_dataset(args.data_path, "valid")
    train_dataloader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    valid_dataloader = DataLoader(
        valid_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    criterion = nn.BCELoss()

    # Weight_decay to avoid overfitting
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=3e-5
    )

    # Warm-up: Growing learning rate linearly start from 1% of learning rate
    warmup_epochs = 10
    cosine_epochs = args.epochs - warmup_epochs

    linear_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)

    cosine_scheduler = CosineAnnealingLR(
        optimizer=optimizer, eta_min=args.learning_rate * 0.1, T_max=cosine_epochs
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[linear_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )

    model.train()

    for epoch in range(args.epochs):

        train_loss_sum = 0
        train_dice_score_sum = 0
        for sample in tqdm(train_dataloader, desc=f"Epoch [{epoch+1}/{args.epochs}]"):

            optimizer.zero_grad()

            # -------------------- Training Phase Begin--------------------
            image, mask = sample["image"].to(device=device), sample["mask"].to(
                device=device
            )

            pred_mask = model(image)
            dice_score_value = dice_score(pred_mask, mask).item()

            loss = 0.5 * criterion(pred_mask, mask) + 1.5 * dice_loss(pred_mask, mask)
            # loss.item() return the loss value
            train_loss_sum += loss.item()
            train_dice_score_sum += dice_score_value

            loss.backward()
            optimizer.step()
            # -------------------- Training Phase End--------------------

        # Calculate the average of training loss and dice score within one epoch
        avg_train_loss = train_loss_sum / len(train_dataloader)
        avg_train_dice = train_dice_score_sum / len(train_dataloader)
        train_loss.append(avg_train_loss)
        train_dice_score.append(avg_train_dice)

        tqdm.write(
            "Epoch {:d} - Training Loss: {:.4f} , Training Dice Score: {:.4f}".format(
                epoch + 1, avg_train_loss, avg_train_dice
            )
        )

        # -------------------- Validating Phase Begin--------------------

        val_loss_value, val_dice_score_value = evaluate(model, valid_dataloader, device)
        val_loss.append(val_loss_value)
        val_dice_score.append(val_dice_score_value)
        scheduler.step()
        lr_history.append(scheduler.get_last_lr()[0])
        # -------------------- Validating Phase End--------------------

        # Save model
        if val_dice_score_value > best_dice_score:
            best_dice_score = val_dice_score_value
            torch.save(model.state_dict(), "saved_models/{}.pth".format(args.model))

    # -------------------- Draw Loss and Dice Curve --------------------

    show_result(
        train_loss, train_dice_score, val_loss, val_dice_score, args.model, lr_history
    )

    if not os.path.exists("saved_metrics"):
        os.mkdir("saved_metrics")
    # Store loss and dice_score array for analyzing purpose
    np.save(f"saved_metrics/{args.model}_train_loss.npy", np.array(train_loss))
    np.save(f"saved_metrics/{args.model}_train_dice.npy", np.array(train_dice_score))
    np.save(f"saved_metrics/{args.model}_val_loss.npy", np.array(val_loss))
    np.save(f"saved_metrics/{args.model}_val_dice.npy", np.array(val_dice_score))


def get_args():
    # python src/train.py --model ResNet34_Unet --epochs 100 --batch_size 8 -lr 1e-4
    parser = argparse.ArgumentParser(
        description="Train the Model on images and target masks"
    )
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
        help="The model for tranining, Unet or ResNet34_Unet",
    )
    parser.add_argument(
        "--epochs", "-e", type=int, default=200, help="number of epochs"
    )
    parser.add_argument("--batch_size", "-b", type=int, default=16, help="batch size")
    parser.add_argument(
        "--learning_rate", "-lr", type=float, default=1e-4, help="learning rate"
    )

    return parser.parse_args()


if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = get_args()
    model = None
    if args.model == "Unet":
        model = Unet(3, 1).to(device=device)
    else:
        model = ResNet34_UNet(3, 1).to(device=device)

    train(args, device, model)
