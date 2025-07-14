import argparse
import torch
from oxford_pet import load_dataset
from torch.utils.data import DataLoader
from models.unet import Unet
from tqdm import tqdm
import numpy as np
from utils import dice_score
from torch import nn

def inference(args, device):
    test_data = load_dataset(args.data_path, mode="test", transform=None)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    test_loss = []
    test_dice_scores = []
    
    # TODO : model selection, if model doesn't exist ,we should train it first , otherwise we load it from file
    model = Unet(3, 1)
    
    criterion = nn.BCELoss()
    model.eval()

    with torch.no_grad():
        for data in test_dataloader:
            image = data["image"].to(device)
            mask = data["mask"].to(device)
            pred_mask = model(image)
            test_loss.append(criterion(pred_mask, mask).item())
            test_dice_scores.append(dice_score(pred_mask, mask).item())

        tqdm.write(
            "Test loss: {:.4f}, Test Dice Score: {:.4f}".format(
                np.mean(test_loss), np.mean(test_dice_scores)
            )
        )
    
    


def get_args():
    parser = argparse.ArgumentParser(description="Predict masks from input images")
    parser.add_argument(
        "--model", "-m", default="U", help="Select model, U: Unet / R: Resnet34_Unet"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="dataset\oxford-iiit-pet",
        help="path to the input data",
    )
    parser.add_argument("--batch_size", "-b", type=int, default=32, help="batch size")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    inference(args)
