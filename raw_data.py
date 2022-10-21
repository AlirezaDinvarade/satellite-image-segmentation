import numpy as np
import os
import rasterio as rio
import cv2
import torch.nn.functional as F
import torch
import splitfolders
from sklearn.decomposition import PCA, IncrementalPCA
import pandas as pd


def Convert_lbl(target):
    target = np.where(target != 10, target, 0)  # Trees
    target = np.where(target != 20, target, 1)  # Shrubland
    target = np.where(target != 30, target, 1)  # Grassland
    target = np.where(target != 40, target, 2)  # Cropland
    target = np.where(target != 50, target, 3)  # Built-up
    target = np.where(target != 60, target, 4)  # Barren ==> As background
    target = np.where(target != 70, target, 5)  # Snow and ice
    target = np.where(target != 80, target, 6)  # Open water
    target = np.where(target != 90, target, 6)  # Herbaceous wetland
    target = np.where(target != 95, target, 0)  # Mangroves
    target = np.where(target != 100, target, 1)  # Moss and lichen
    return target


def uint8(img):
    image = (img/img.max())*255
    image = np.uint8(image)
    return image


def pca_process(data , channel_num = 16):

    if channel_num == 16:
        pca = PCA()
        data = data.transpose(1, 2, 0)

        first, second, third, fourth, fifth, sixth, seventh, eighth, ninth, tenth, eleventh, towelth, thirteenth = cv2.split(
            data)
        a = pca.fit_transform(first)
        # Evaluate variance
        var_cumu = np.cumsum(pca.explained_variance_ratio_) * 100
        # How many PCs explain 95% of the variance?
        k = np.argmax(var_cumu > 95)
        k = k + 50
        # print("Number of components explaining 95% variance: "+ str(k))
        ipca = IncrementalPCA(n_components=k)
        image = np.stack((uint8(first), uint8(second), uint8(third), uint8(fourth), uint8(fifth), uint8(sixth), uint8(seventh),
                          uint8(eighth), uint8(ninth), uint8(tenth), uint8(eleventh), uint8(towelth), uint8(thirteenth)))

    else:
        pca = PCA()
        data = data.transpose(1, 2, 0)
        first, second, third, fourth, fifth, sixth, seventh, eighth, ninth, tenth, eleventh, towelth, thirteenth = cv2.split(
            data)
        a = pca.fit_transform(first)
        # Evaluate variance
        var_cumu = np.cumsum(pca.explained_variance_ratio_) * 100
        # How many PCs explain 95% of the variance?
        k = np.argmax(var_cumu > 95)
        k = k + 50
        # print("Number of components explaining 95% variance: "+ str(k))
        ipca = IncrementalPCA(n_components=k)
        a_re = ipca.inverse_transform(ipca.fit_transform(first))
        b_re = ipca.inverse_transform(ipca.fit_transform(second))
        c_re = ipca.inverse_transform(ipca.fit_transform(third))
        image = np.stack((a_re, b_re, c_re))

    return image


def get_lalbel_address(names):
    idx = names.replace("_", " ")
    idx = idx.replace(".", " ")
    idx = idx.replace("/", " ")
    idx = idx.split()
    index = idx[-2]
    root = idx[:-4]
    return index, root


def get_index(names):
    idx = names.replace("_", " ")
    idx = idx.replace(".", " ")
    idx = idx.replace("/", " ")
    idx = idx.split()
    return idx


def save_label(indx, img, transform):
    new_dataset = rio.open(f"label{indx}.tif", 'w', driver='GTiff', height=512, width=512,
                           count=1, dtype=str(img.dtype),
                           crs='+proj=utm +zone=10 +ellps=GRS80 +datum=NAD83 +units=m +no_defs',
                           transform=transform)
    new_dataset.write(img)
    new_dataset.close()


def save_img(indx, img, transform):
    new_dataset = rio.open(f"sentinel2_{indx}.tif", 'w', driver='GTiff', height=img.shape[1], width=img.shape[2],
                           count=13, dtype=str(img.dtype),
                           crs='+proj=utm +zone=10 +ellps=GRS80 +datum=NAD83 +units=m +no_defs',
                           transform=transform)
    new_dataset.write(img)
    new_dataset.close()


def interpolate(img):
    img = torch.from_numpy(img)
    img = F.interpolate(input=img[None, :], size=(512, 512), mode='nearest-exact')
    return img.squeeze().numpy()


data_path = []
sent_path = []
label_path = []

'Path to DATASET to Apply PCA'
for root, dirs, files in os.walk("/home/hamtech/Desktop/2T/home/sedreh/seg/data/rawdata", topdown=False):
    for i in range(len(files)):
        address = root + '/' + files[i]
        data_path.append(address)
#
for i in range(len(data_path)):
    id = get_index(data_path[i])
    if 'sentinel2' in id:
        sentinel = data_path[i]
        idx, root = get_lalbel_address(sentinel)
        label_address = '/'.join(root)
        label_address = f'/{label_address}/lable/lable{idx}.tif'
        sent_path.append(sentinel)
        label_path.append(label_address)
        print(i, sentinel, '====', label_address)
print('DATA Amout: ', len(sent_path))

'Last image index which had been loaded before'
index = int(input('Enter last image index that applied: '))
bug_data = '/home/hamtech/Desktop/2T/home/sedreh/seg/bug_data.txt'
for names in range(len(sent_path)):
    try :
        sent = rio.open(sent_path[names])
        img = sent.read()
        transform_s = sent.transform
        sent = []
        'calculates pcs` and indices values'
        img = pca_process(img)
        h, w = img.shape[1], img.shape[2]
        label = rio.open(label_path[names])
        'to rename images, each pol has N images, so renaming is needed'
        transform_l = label.transform
        label = label.read()
        label = label[:, :h, :w]
        img = interpolate(img)
        label = interpolate(label)
        label = Convert_lbl(label)
        'Save results'
        os.chdir('/home/hamtech/Desktop/2T/home/sedreh/seg/data/resized/masks')
        save_label(index, label[None, :], transform_l)
        os.chdir('/home/hamtech/Desktop/2T/home/sedreh/seg/data/resized/images')
        save_img(index, img, transform_s)
        index += 1
    except :
        print('='*50)
        print(f'couldnt apply preprocess to {sent_path[names]}')
        with open(bug_data, 'a') as bd:
            bd.write("%s\n" % sent_path[names])
        pass

input_folder = '/home/hamtech/Desktop/2T/home/sedreh/seg/data/resized'
output_folder = '/home/hamtech/Desktop/2T/home/sedreh/seg/data/split/'
splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(.9, .05 , .05), group_prefix=None)


train_image_path = []
for root, dirs, files in os.walk("/home/hamtech/Desktop/2T/home/sedreh/seg/data/split/train/images", topdown=True):
    for i in sorted(files):
        address = root + '/' + i
        train_image_path.append(address)

train_mask_path = []
for root, dirs, files in os.walk("/home/hamtech/Desktop/2T/home/sedreh/seg/data/split/train/masks", topdown=True):
    for i in sorted(files):
        address = root + '/' + i
        train_mask_path.append(address)

val_image_path = []
for root, dirs, files in os.walk("/home/hamtech/Desktop/2T/home/sedreh/seg/data/split/val/images", topdown=True):
    for i in sorted(files):
        address = root + '/' + i
        val_image_path.append(address)

val_mask_path = []
for root, dirs, files in os.walk("/home/hamtech/Desktop/2T/home/sedreh/seg/data/split/val/masks", topdown=True):
    for i in sorted(files):
        address = root + '/' + i
        val_mask_path.append(address)

test_image_path = []
for root, dirs, files in os.walk("/home/hamtech/Desktop/2T/home/sedreh/seg/data/split/test/images", topdown=True):
    for i in sorted(files):
        address = root + '/' + i
        test_image_path.append(address)

test_mask_path = []
for root, dirs, files in os.walk("/home/hamtech/Desktop/2T/home/sedreh/seg/data/split/test/masks", topdown=True):
    for i in sorted(files):
        address = root + '/' + i
        test_mask_path.append(address)

train = {'col1': train_image_path, 'col2': train_mask_path}
train = pd.DataFrame(data=train)

val = {'col1': val_image_path, 'col2': val_mask_path}
val = pd.DataFrame(data=val)

test = {'col1': test_image_path, 'col2': test_mask_path}
test = pd.DataFrame(data=test)

train.to_csv("/home/hamtech/Desktop/2T/home/sedreh/seg/train.csv")
val.to_csv("/home/hamtech/Desktop/2T/home/sedreh/seg/val.csv")
test.to_csv("/home/hamtech/Desktop/2T/home/sedreh/seg/test.csv")









