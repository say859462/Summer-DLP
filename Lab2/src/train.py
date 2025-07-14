import argparse
from oxford_pet import load_dataset
from torch.utils.data import DataLoader
from models.unet import Unet
import torch
from torch import nn
from tqdm import tqdm
import numpy as np
from utils import dice_score
from evaluate import evaluate
import albumentations as A

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Data augmentation
# Include rotation,and adjustment of brightness, contrast , saturation, hue
def train_transform():
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),
        ]
    )


# Plot the result
def show_result(train_loss, train_dice_score, val_loss, val_dice_score):
    import matplotlib.pyplot as plt

    # Loss Curve
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label="Train Loss", color="blue", marker="o")
    plt.plot(val_loss, label="Validation Loss", color="orange", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True)

    # Save loss curve
    plt.savefig("loss_curve.png", bbox_inches="tight", dpi=300)
    plt.close()

    # Dice Score curve
    plt.figure(figsize=(10, 5))
    plt.plot(train_dice_score, label="Train Dice", color="blue", marker="o")
    plt.plot(val_dice_score, label="Validation Dice", color="orange", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Dice Score")
    plt.title("Training & Validation Dice Score")
    plt.legend()
    plt.grid(True)

    # Save dice score curve
    plt.savefig("dice_curve.png", bbox_inches="tight", dpi=300)
    plt.close()


def train(args):
    # implement the training function here

    train_loss = []
    train_dice_score = []
    val_loss = []
    val_dice_score = []
    best_dice_score = 0.0

    train_data = load_dataset(args.data_path, "train", transform=train_transform())
    valid_data = load_dataset(args.data_path, "valid")
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False)
    model = Unet(3, 1).to(device)

    # We choose Binary Cross Entropy due to the output of model (foreground and background)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    model.train()
    for epoch in range(args.epochs):

        train_loss_sum = 0
        train_dice_score_sum = 0
        for sample in tqdm(train_dataloader, desc=f"Epoch [{epoch+1}/{args.epochs}]"):

            optimizer.zero_grad()

            # -------------------- Training Phase Begin--------------------
            image, mask = sample["image"].to(device), sample["mask"].to(device)

            pred_mask = model(image)
            dice = dice_score(pred_mask, mask)
            loss = criterion(pred_mask, mask)

            # loss.item() return the loss value
            train_loss_sum += loss.item()
            train_dice_score_sum += dice.item()

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

        # -------------------- Validating Phase End--------------------

        # Save model
        if val_dice_score_value > best_dice_score:
            best_dice_score = val_dice_score_value

            torch.save(model.state_dict(), f"saved_models/Unet.pth")

    # -------------------- Draw Loss and Dice Curve --------------------
    show_result(train_loss, train_dice_score, val_loss, val_dice_score)


def get_args():
    parser = argparse.ArgumentParser(
        description="Train the UNet on images and target masks"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="dataset\oxford-iiit-pet",
        help="path of the input data",
    )
    parser.add_argument(
        "--epochs", "-e", type=int, default=100, help="number of epochs"
    )
    parser.add_argument("--batch_size", "-b", type=int, default=32, help="batch size")
    parser.add_argument(
        "--learning_rate", "-lr", type=float, default=1e-3, help="learning rate"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    train(args)
