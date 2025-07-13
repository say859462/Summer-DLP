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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(args):
    # implement the training function here

    train_loss = []
    val_loss = []
    val_dice_score = []

    best_dice_score = None

    train_data = load_dataset(args.data_path, "train")
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
        for sample in tqdm(train_dataloader):

            # -------------------- Training Phase Begin--------------------
            image, mask = sample["image"].to(device), sample["mask"].to(device)

            pred_mask = model(image)
            dice_score(pred_mask, mask)
            loss = criterion(pred_mask, mask)

            # loss.item() return the loss value
            train_loss_sum += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # -------------------- Training Phase End--------------------

        train_loss.append(train_loss_sum / len(train_dataloader))
        tqdm.write("Epoch {}, Loss: {:.4f} ".format(epoch + 1, train_loss[-1]))

        # -------------------- Validating Phase Begin--------------------
        with torch.no_grad():
            val_loss_value, val_dice_score_value = evaluate(
                model, valid_dataloader, device
            )
            val_loss.append(val_dice_score_value)
            val_dice_score.append(val_dice_score_value)

        # -------------------- Validating Phase End--------------------

        # Save model
        if (best_dice_score is None) or (val_dice_score < best_dice_score):
            best_dice_score = val_loss
            torch.save(model.state_dict(), f"/saved_models/Unet.pth")

    # -------------------- Draw Loss and Dice Curve --------------------
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(train_loss, label="Train Loss", color="blue")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(val_loss, label="Validation Loss", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Validation Loss")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(val_dice_score, label="Validation Dice Score", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Dice Score")
    plt.title("Validation Dice Score")
    plt.legend()

    plt.tight_layout()
    plt.show()


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
    parser.add_argument("--epochs", "-e", type=int, default=5, help="number of epochs")
    parser.add_argument("--batch_size", "-b", type=int, default=32, help="batch size")
    parser.add_argument(
        "--learning_rate", "-lr", type=float, default=1e-3, help="learning rate"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    train(args)
