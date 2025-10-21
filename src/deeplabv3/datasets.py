import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import albumentations as A
import glob

from src.deeplabv3.utils import set_class_labels, get_label_mask


def get_images(path):
    train_imgs = glob.glob(f"{path}/document_dataset_resized/train/images/*")
    train_masks = glob.glob(f"{path}/document_dataset_resized/train/masks/*")
    valid_imgs = glob.glob(f"{path}/document_dataset_resized/valid/images/*")
    valid_masks = glob.glob(f"{path}/document_dataset_resized/valid/masks/*")

    return train_imgs, train_masks, valid_imgs, valid_masks


# normalizing before passing in to train  the model, might have to
# tune it a bit for our dataset, but for now just leave it
def normalize():
    """
    Transform to normalize image.
    """
    transform = A.Compose([
        A.Normalize(
            mean=[0.45734706, 0.43338275, 0.40058118],
            std=[0.23965294, 0.23532275, 0.2398498],
            always_apply=True
        )
    ])
    return transform

# right now we only resize, flip image, and have random brightness,
# but we should add transformations for other stuff (figure it out later)
# actually, if we use a synthetic dataset, we wouldn't need to transform
# if that dataset is already transformed. but we could just use a synthetic
# dataset and transform on the fly with this, not sure
def train_transforms(img_size):
    """
    Transforms/augmentations for training images and masks.

    :param img_size: Integer, for image resize.
    """
    train_image_transform = A.Compose([
        A.Resize(img_size, img_size, always_apply=True),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
    ])
    return train_image_transform

# only transofmrmations should be resizing, why? idk figure it out later
def valid_transforms(img_size):
    """
    Transforms/augmentations for validation images and masks.

    :param img_size: Integer, for image resize.
    """
    valid_image_transform = A.Compose([
        A.Resize(img_size, img_size, always_apply=True),
    ])
    return valid_image_transform

class SegmentationDataset(Dataset):
    # ctor
    def __init__(
            self, image_paths, mask_paths, tfms, norm_tfms, label_colors_list,classes_to_train,all_classes
    ):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        # transformation functions
        self.tfms = tfms
        self.norm_tfms = norm_tfms
        self.label_colors_list = label_colors_list
        self.all_classes = all_classes
        self.classes_to_train = classes_to_train
        # Convert string names to class values for masks.
        self.class_values = set_class_labels(
            self.all_classes, self.classes_to_train
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = np.array(Image.open(self.image_paths[index]).convert('RGB'))
        mask = np.array(Image.open(self.mask_paths[index]).convert('RGB'))

        # prob is not necessary but ok
        im = mask >= 200
        mask[im] = 255
        mask[np.logical_not(im)] = 0


        image = self.norm_tfms(image=image)['image']
        transformed = self.tfms(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask']

        # Get colored label mask.
        mask = get_label_mask(mask, self.class_values, self.label_colors_list)

        # transpose from (height, width, channel) to (C, H, W)
        # since this is the format that deeplabv3 expects in torch
        image = np.transpose(image, (2, 0, 1))

        image = torch.tensor(image, dtype=torch.float)
        mask = torch.tensor(mask, dtype=torch.long)

        return image, mask


def get_dataset(
    train_image_paths,
    train_mask_paths,
    valid_image_paths,
    valid_mask_paths,
    all_classes,
    classes_to_train,
    label_colors_list,
    img_size
):
    train_tfms = train_transforms(img_size)
    valid_tfms = valid_transforms(img_size)
    norm_tfms = normalize()

    train_dataset = SegmentationDataset(
        train_image_paths,
        train_mask_paths,
        train_tfms,
        norm_tfms,
        label_colors_list,
        classes_to_train,
        all_classes
    )
    valid_dataset = SegmentationDataset(
        valid_image_paths,
        valid_mask_paths,
        valid_tfms,
        norm_tfms,
        label_colors_list,
        classes_to_train,
        all_classes
    )
    return train_dataset, valid_dataset
