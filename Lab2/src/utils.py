import torch
import torch.nn
from PIL import Image
import numpy as np


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


def show_maskOnimage(pred_mask, image, id, args):
    pred_mask = pred_mask.squeeze(0).squeeze(0).cpu().numpy()

    binary_mask = (pred_mask > 0.5).astype(np.uint8)

    np_image = (image.squeeze(0).permute(1, 2, 0).cpu().numpy()).astype(np.uint8)
    pil_image = Image.fromarray(np_image).convert("RGBA")

    np_rgba = np.array(pil_image)

    # lower alpha value of foreground
    # foreground：alpha=120；background：alpha=255
    np_rgba[..., 3] = np.where(binary_mask == 1, 120, 255)
    pil_result = Image.fromarray(np_rgba)
    pil_result.save(f"outputs_imgs/stack/{args.model}/{id}.png")


def show_maskAndPredMask(pred_mask, mask, id, args):
    pred_mask = pred_mask.squeeze(0).squeeze(0).cpu().numpy()
    mask = mask.squeeze(0).squeeze(0).cpu().numpy()

    binary_pred_mask = (pred_mask > 0.5).astype(np.uint8)
    binary_mask = (mask > 0.5).astype(np.uint8)

    # Create RGBA images for true mask and predicted mask
    height, width = binary_pred_mask.shape
    rgba_true_mask = np.zeros((height, width, 4), dtype=np.uint8)
    rgba_pred_mask = np.zeros((height, width, 4), dtype=np.uint8)

    # Assign colors: green for true mask, red for predicted mask
    rgba_true_mask[..., 1] = np.where(binary_mask == 1, 255, 0)  # Green for true mask
    rgba_true_mask[..., 3] = np.where(binary_mask == 1, 120, 255)  # Alpha channel
    rgba_pred_mask[..., 0] = np.where(
        binary_pred_mask == 1, 255, 0
    )  # Red for predicted mask
    rgba_pred_mask[..., 3] = np.where(binary_pred_mask == 1, 120, 255)  # Alpha channel

    # Convert to PIL images
    pil_true_mask = Image.fromarray(rgba_true_mask)
    pil_pred_mask = Image.fromarray(rgba_pred_mask)

    # Add titles using PIL
    from PIL import ImageDraw

    draw_true = ImageDraw.Draw(pil_true_mask)
    draw_pred = ImageDraw.Draw(pil_pred_mask)
    title_true = "True Mask"
    title_pred = "Predicted Mask"
    draw_true.text(
        (10, 10), title_true, fill=(0, 255, 0, 255)
    )  # Green text for true mask
    draw_pred.text(
        (10, 10), title_pred, fill=(255, 0, 0, 255)
    )  # Red text for predicted mask

    # Combine images side by side
    combined_width = width * 2
    combined_height = height
    combined_image = Image.new("RGBA", (combined_width, combined_height))
    combined_image.paste(pil_true_mask, (0, 0))
    combined_image.paste(pil_pred_mask, (width, 0))

    # Add overall title to the combined image
    draw_combined = ImageDraw.Draw(combined_image)
    draw_combined.text(
        (10, 10), "True Mask | Predicted Mask", fill=(255, 255, 255, 255)
    )

    # Save the combined image
    combined_image.save(f"outputs_imgs/combined/{args.model}/combined_mask_{id}.png")
