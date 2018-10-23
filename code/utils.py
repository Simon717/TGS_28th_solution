import os
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from metric import do_kaggle_metric
from config import *



def mkdir(paths):
    """make a path or several paths"""
    if not isinstance(paths, (list, tuple)):
        paths = [paths]
    for path in paths:
        path_dir, _ = os.path.split(path)
        if not os.path.isdir(path_dir):
            os.makedirs(path_dir)

def read_ids(path):
    with open(path, 'r') as f:
        out = []
        for line in f.readlines():
            out.append(line.strip('\n'))
        return out

def mkdir_outputs(STAGE_ID, subfolder_dir, debug):
    # log file
    dir_log = ('../DEBUG/' if debug else '../') + subfolder_dir + 'log/' + 'stage{}/'.format(STAGE_ID)
    mkdir(dir_log)

    # models file
    dir_models = ('../DEBUG/' if debug else '../') + subfolder_dir  + 'models/'+ 'stage{}/'.format(STAGE_ID)
    mkdir(dir_models)

    # subs file
    dir_subs = ('../DEBUG/' if debug else '../') + subfolder_dir + 'subs/' + 'stage{}/'.format(STAGE_ID)
    mkdir(dir_subs)
    return dir_log, dir_models, dir_subs

def rle_encode(im):
    pixels = im.flatten(order='F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)




def de_pad(img_array):
    return img_array[:,:,Y0:Y1, X0:X1]

def de_pad2(image, mask):
    return de_pad(image), de_pad(mask)
# def down_sample(img)

def up_sample(img, target_size=(224,224)):
    return cv2.resize(img, target_size)
def upsample_array(img):
    return np.array([up_sample(x) for x in img])

def down_sample(img, target_size=(101, 101)):
    return cv2.resize(img, target_size)
def down_sample_array(img):
    return np.array([down_sample(x) for x in img])

# def filter_image(img):
#     if img.sum() < 100:
#         return np.zeros(img.shape)
#     else:
#         return img

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def do_valid(model, dataloader, device, is_lova=False):
    valid_num = 0
    valid_loss = np.zeros(3, np.float32)

    predicts = []
    truths = []

    model.set_mode('valid')
    for image, mask in dataloader:
        image, mask = image.to(device), mask.to(device)
        with torch.no_grad():
            logit = model(image)
            prob = torch.sigmoid(logit)
            loss = model.criterion(logit, mask, is_lova=is_lova)
            dice = model.dice(logit, mask)

        batch_size = image.shape[0]  # TODO
        valid_loss += batch_size * np.array((loss.item(), dice.item(), 0))
        valid_num += batch_size

        # unAUG
        prob, mask = de_pad2(prob, mask)
        prob = F.avg_pool2d(prob, kernel_size=2, stride=2)
        mask = F.avg_pool2d(mask, kernel_size=2, stride=2)
        mask = (mask > 0.5).float()

        predicts.append(prob.data.cpu().numpy())
        truths.append(mask.data.cpu().numpy())

    assert (valid_num == len(dataloader.sampler))
    valid_loss = valid_loss / valid_num

    predicts = np.concatenate(predicts).squeeze()
    truths = np.concatenate(truths).squeeze()
    # predicts = np.array([down_sample(x) for x in predicts])
    # truths = np.array([down_sample(x) for x in truths])
    precision, result, threshold = do_kaggle_metric(predicts, truths)
    valid_loss[2] = precision.mean()

    return valid_loss
