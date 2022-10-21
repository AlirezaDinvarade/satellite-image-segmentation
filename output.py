import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
import rasterio as rio
from PIL import Image
import albumentations as A
os.chdir("/home/hamtech/Desktop/2T/home/sedreh/seg")
train = pd.read_csv("/home/hamtech/Desktop/2T/home/sedreh/seg/train.csv")
test = pd.read_csv("/home/hamtech/Desktop/2T/home/sedreh/seg/val.csv")


MULTICLASS_MODE: str = "multiclass"
ENCODER = "resnet18"
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['trees', 'grassland', 'cropland', 'built_up','barren','snow','open_water']
ACTIVATION = None
DEVICE = 'cuda'


class MultiClassSegDataset(Dataset):

    def __init__(self, df, classes=None, transform=None, ):
        self.df = df
        self.classes = classes
        self.transform = transform

    def __getitem__(self, idx):

        image_name = self.df.iloc[idx, 1]
        mask_name = self.df.iloc[idx, 2]
        img = rio.open(image_name)
        image = img.read()
        image = image.transpose(1,2,0)
        msk = rio.open(mask_name)
        mask = msk.read()
        mask = mask.transpose(1,2,0)
        if (self.transform is not None):
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
            image = torch.from_numpy(image)
            mask = torch.from_numpy(mask)
            image = image.permute(2,0,1)
            image = image.float()
            mask = mask.permute(2,0,1)
            mask = mask.long()
        else:
            image = torch.from_numpy(image)
            mask = torch.from_numpy(mask)
            image = image.permute(2,0,1)
            mask = mask.permute(2,0,1)
            image = image.float()
            mask = mask.long()
        return image, mask

    def __len__(self):
        return len(self.df)


train_transform = A.Compose(
    [   A.ToFloat(max_value=256),
        A.Resize(512, 512),
    ]
)


best_model = torch.load('./model_epoch84.pth')

# Evaluate model on validation set==============================
val = pd.read_csv("/home/hamtech/Desktop/2T/home/sedreh/seg/val.csv")

valDS = MultiClassSegDataset(val, classes=CLASSES, transform=train_transform)


# Visualize images, masks, and predictions=======================================
def visualize(**images):
    n = len(images)
    plt.figure(figsize=(16, 10))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()


acc =[]
for i in range(10):
    n = np.random.choice(len(valDS))
    image_vis = valDS[n][0].permute(1,2,0)
    image_vis = image_vis.numpy()
    image_vis = image_vis.astype('uint8')
    image_vis = image_vis[: , : , 0:3]
    image, gt_mask = valDS[n]
    gt_mask = gt_mask.squeeze()
    x_tensor = image.to(DEVICE).unsqueeze(0)
    pr_mask = best_model(x_tensor)
    m = nn.Softmax(dim=1)
    pr_probs = m(pr_mask)
    pr_mask = torch.argmax(pr_probs, dim=1).squeeze(1)
    pr_mask = pr_mask.squeeze().cpu()

    visualize(
        image=image_vis,
        ground_truth_mask=gt_mask,
        predicted_mask=pr_mask
    )
