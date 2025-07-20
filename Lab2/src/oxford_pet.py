import os
import shutil
from urllib.request import urlretrieve

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


class OxfordPetDataset(torch.utils.data.Dataset):

    def __init__(self, root, mode="train", transform=None):
        """_summary_

        Args:
            root (_type_): Dataset location
            mode (str, optional): Training or Validation Set, Defaults to "train".
            transform (_type_, optional): Applying data transform(augmentation) for dataset, Defaults to None.
        """
        assert mode in {"train", "valid", "test"}

        self.root = root
        self.mode = mode
        self.transform = transform

        self.images_directory = os.path.join(
            self.root, "images"
        )  # images_directory = root/images
        self.masks_directory = os.path.join(self.root, "annotations", "trimaps")
        # Check whether the dataset is exists , otherwise,we should download the dataset first
        if not os.path.exists(self.images_directory) or not os.path.exists(
            self.masks_directory
        ):
            self.download(root)

        self.filenames = self._read_split()  # read train/valid/test splits

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):

        filename = self.filenames[idx]
        image_path = os.path.join(self.images_directory, filename + ".jpg")
        mask_path = os.path.join(self.masks_directory, filename + ".png")

        image = np.array(Image.open(image_path).convert("RGB"))

        trimap = np.array(Image.open(mask_path))
        mask = self._preprocess_mask(trimap)

        sample = dict(image=image, mask=mask, trimap=trimap)

        # We transform the data only on training data
        if self.transform is not None and self.mode == "train":
            sample = self.transform(**sample)

        return sample

    @staticmethod
    def _preprocess_mask(mask):
        mask = mask.astype(np.float32)
        mask[mask == 2.0] = 0.0

        mask[(mask == 1.0) | (mask == 3.0)] = 1.0
        return mask

    def _read_split(self):
        split_filename = "test.txt" if self.mode == "test" else "trainval.txt"
        split_filepath = os.path.join(self.root, "annotations", split_filename)

        with open(split_filepath) as f:
            split_data = f.read().strip("\n").split("\n")
        filenames = [x.split(" ")[0] for x in split_data]
        if self.mode == "train":  # 90% for train
            filenames = [x for i, x in enumerate(filenames) if i % 10 != 0]
        elif self.mode == "valid":  # 10% for validation
            filenames = [x for i, x in enumerate(filenames) if i % 10 == 0]
        return filenames

    @staticmethod
    def download(root):

        # load images
        filepath = os.path.join(root, "images.tar.gz")
        download_url(
            url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
            filepath=filepath,
        )
        extract_archive(filepath)

        # load annotations
        filepath = os.path.join(root, "annotations.tar.gz")
        download_url(
            url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz",
            filepath=filepath,
        )
        extract_archive(filepath)


class SimpleOxfordPetDataset(OxfordPetDataset):

    def __getitem__(self, *args, **kwargs):

        sample = super().__getitem__(*args, **kwargs)
        # resize images
        image = np.array(
            Image.fromarray(sample["image"]).resize((256, 256), Image.BILINEAR)
        )
        mask = np.array(
            Image.fromarray(sample["mask"]).resize((256, 256), Image.NEAREST)
        )
        trimap = np.array(
            Image.fromarray(sample["trimap"]).resize((256, 256), Image.NEAREST)
        )

        # convert to other format HWC -> CHW
        sample["image"] = np.moveaxis(image, -1, 0).astype(np.float32)
        sample["mask"] = np.expand_dims(mask, 0).astype(np.float32)
        sample["trimap"] = np.expand_dims(trimap, 0)

        return sample


class TqdmUpTo(tqdm):

    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, filepath):
    directory = os.path.dirname(os.path.abspath(filepath))
    os.makedirs(directory, exist_ok=True)

    # If we already downloaded the dataset , we should not download it again
    if os.path.exists(filepath):
        return

    with TqdmUpTo(
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        miniters=1,
        desc=os.path.basename(filepath),
    ) as t:
        urlretrieve(url, filename=filepath, reporthook=t.update_to, data=None)
        t.total = t.n


def extract_archive(filepath):
    extract_dir = os.path.dirname(os.path.abspath(filepath))
    dst_dir = os.path.splitext(filepath)[0]
    if not os.path.exists(dst_dir):
        shutil.unpack_archive(filepath, extract_dir)


import albumentations as A


def load_dataset(data_path="dataset\oxford-iiit-pet", mode="train"):
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
                A.RandomBrightnessContrast(
                    brightness_limit=0.3, contrast_limit=0.3, p=0.5
                ),
            ],
            additional_targets={"mask": "mask"},
        )

    return SimpleOxfordPetDataset(data_path, mode=mode, transform=train_transform())


# For local-develope purpose
if __name__ == "__main__":
    # Data augmentation
    # Include rotation,and adjustment of brightness, contrast , saturation, hue=
    def train_transform():
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5
                ),
            ]
        )

    data = load_dataset(
        "dataset\oxford-iiit-pet", mode="train", transform=train_transform()
    )
