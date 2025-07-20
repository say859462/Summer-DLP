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
from utils import dice_score, dice_loss


# Data augmentation
# additional_targets indicate that what kind of data should apply image augmentation and what should not
# e.g : image data apply blur is available but mask data should not

# CLANE : edge enhancement , benefit to learn edge of foreground


def train_transform():
    return A.Compose(
        [
            A.RandomResizedCrop(
                size=[256, 256], scale=[0.8, 1.0], ratio=[0.75, 1.33], p=1.0
            ),
            A.HorizontalFlip(p=0.5),
            A.HueSaturationValue(
                hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5
            ),
            A.OneOf(
                [
                    A.GridDistortion(num_steps=5, distort_limit=0.3),
                    A.CoarseDropout(
                        num_holes_range=[1, 2],
                        hole_height_range=[0.1, 0.2],
                        hole_width_range=[0.1, 0.2],
                        fill=0,
                    ),
                    A.Affine(
                        translate_percent=0.2,
                        scale=(0.7, 1.3),
                        rotate=(-45, 45),
                        shear=(-5, 5),
                    ),
                ],
                p=0.5,
            ),
            A.OneOf(
                [
                    A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8)),
                    A.Blur(blur_limit=3),
                ],
                p=0.5,
            ),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        ],
        additional_targets={"mask": "mask"},
    )


# Plot the result
def show_result(
    train_loss, train_dice_score, val_loss, val_dice_score, model_name, lr_history=None
):
    import matplotlib.pyplot as plt

    min_train_loss = min(train_loss)
    min_val_loss = min(val_loss)
    max_train_dice = max(train_dice_score)
    max_val_dice = max(val_dice_score)

    # Loss Curve
    plt.figure(figsize=(10, 5))
    plt.plot(
        train_loss,
        label="Train Loss (min: {:.4f})".format(min_train_loss),
        color="blue",
    )
    plt.plot(
        val_loss,
        label="Validation Loss (min: {:.4f})".format(min_val_loss),
        color="orange",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True)

    # Mark the lowest loss point
    plt.scatter(
        train_loss.index(min_train_loss), min_train_loss, color="blue", s=100, zorder=5
    )
    plt.scatter(
        val_loss.index(min_val_loss), min_val_loss, color="orange", s=100, zorder=5
    )

    # Save loss curve
    plt.savefig(model_name + "_loss_curve.png", bbox_inches="tight", dpi=300)
    plt.close()

    # Dice Score curve
    plt.figure(figsize=(10, 5))
    plt.plot(
        train_dice_score,
        label="Train Dice (max: {:.4f})".format(max_train_dice),
        color="blue",
    )
    plt.plot(
        val_dice_score,
        label="Validation Dice (max: {:.4f})".format(max_val_dice),
        color="orange",
    )

    # Mark the highest dice score point
    plt.scatter(
        train_dice_score.index(max_train_dice),
        max_train_dice,
        color="blue",
        s=100,
        zorder=5,
    )
    plt.scatter(
        val_dice_score.index(max_val_dice),
        max_val_dice,
        color="orange",
        s=100,
        zorder=5,
    )
    plt.xlabel("Epoch")
    plt.ylabel("Dice Score")
    plt.title("Training & Validation Dice Score")
    plt.legend()
    plt.grid(True)

    # Save dice score curve
    plt.savefig(model_name + "_dice_curve.png", bbox_inches="tight", dpi=300)
    plt.close()

    # Learning Rate Curve
    plt.figure(figsize=(10, 5))
    plt.plot(lr_history, label="Learning Rate", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.grid(True)
    plt.legend()
    plt.savefig(model_name + "_lr_curve.png", bbox_inches="tight", dpi=300)
    plt.close()


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

    train_data = load_dataset(args.data_path, "train", transform=train_transform())
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
