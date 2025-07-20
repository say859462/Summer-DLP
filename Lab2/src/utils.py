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