from torch.utils import data
import time
import datetime
from math import cos, pi
from datasetTGS import TGSSaltDataset
from metric import *
from utils import *
from models import Hy_UNetResNet34, Hy_EN_UNetResNet34

UNetResNet = [Hy_UNetResNet34, Hy_EN_UNetResNet34]
UNetResNetstr = ['UNET_', 'SE_UNET']

""" ==================== Global Config ======================== """

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--fold_id", default=0, help="train_valid_fold_id")
parser.add_argument("-n", "--net", default=1, help="unet resnet id")
parser.add_argument("-d", "--depth", default=1, help=" use depth information")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 自动寻找计算设备

data_src = '/home/simon/code/20180921/data/'
torch_pretrain_path = '/home/simon/code/20180921/models/resnet34-333f7ec4.pth'

train_path = data_src + 'train'
test_path = data_src + 'test'

## IDs
fold_id = int(args.fold_id)
net_id = int(args.net)
time_id = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

# if use depth information
DEPTH = int(args.depth)
DEPTH_str = 'Depth_' if DEPTH == 1  else ''
use_depth = True if DEPTH == 1  else False

TTA = True

n_epochs_stage1 = 30
n_epochs_stage2 = 80

# snapshot
cycles = 6
n_epochs_stage3 = 300

SubFolder = True
subfolder_dir = 'DEBUG/' if SubFolder else ''

#=================== DEBUG ====================#

DEBUG = False         # put all output file to ../DEBUG/ you can delete them after the code is correct
DEBUG_mb = False      # fast train: only train on  a minibatch then break the training

""" ==================== Data ======================== """
dir_split_data = '../split_data/'

tr_ids = read_ids(dir_split_data + 'train_fold_{}.txt'.format(fold_id))
valid_ids = read_ids(dir_split_data + 'valid_fold_{}.txt'.format(fold_id))
test_ids = read_ids(dir_split_data + 'test.txt')

dataset_train = TGSSaltDataset(train_path, tr_ids, divide=True, aug='complex', shape_mode='resize', depth=use_depth)
dataset_val = TGSSaltDataset(train_path, valid_ids, divide=True, mode='valid', shape_mode='resize', depth=use_depth)
dataset_test = TGSSaltDataset(test_path, test_ids, divide=True, mode='test', shape_mode='resize', depth=use_depth)

batch_size = 12
train_loader = data.DataLoader(dataset_train,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=8,
                                pin_memory=True)

# Do not shuffle for validation and test.
valid_loader = data.DataLoader(dataset_val,
                                batch_size=batch_size//2,
                                shuffle=False,
                                num_workers=8,
                                pin_memory=True)

test_loader = data.DataLoader(dataset_test,
                                batch_size=batch_size//2,
                                shuffle=False,
                                num_workers=8,
                                pin_memory=True)

""" ==================== Model ======================== """
selected_net = UNetResNet[net_id]

model = selected_net(depth=use_depth).cuda()
model.set_mode('train')

model.load_pretrain(torch_pretrain_path)

""" ==================== Stage 1 ======================== """

#== training params ============

STAGE_ID = 1
tag = DEPTH_str + 'net_{}stage{}_fold_{}_'.format(UNetResNetstr[net_id], STAGE_ID, fold_id)  # net type && fold id  'D04_' +

#== config outputs files ==========
dir_log, dir_models, dir_subs = mkdir_outputs(STAGE_ID=STAGE_ID, subfolder_dir=subfolder_dir, debug=DEBUG)

from logger import Logger
log_fn = dir_log + tag + time_id + '.txt'
print(log_fn)
log = Logger(log_fn)

fn_subs = dir_subs + tag + 'submission_' + time_id + '.csv'
log.write(fn_subs)

log.write('logging from script: ' + sys.argv[0] + '\n') # log the running file name

name_model_best_val_s1 = dir_models + tag + 'best_val_iou_resnet34_' + time_id + '.ckpt'

#============= training =============#

optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=0.01, momentum=0.9, weight_decay=0.0001)

best_val_iou = 0
for e in range(n_epochs_stage1):

    e_t0 = time.time()
    # Training:
    train_loss = []
    train_dice = []
    model.set_mode('train')
    for image, mask in (train_loader):
        image, mask = image.to(device), mask.to(device)

        if torch.cuda.device_count() > 1:
            logit = nn.parallel.data_parallel(model, image)
        else:
            logit = model(image)

        loss = model.criterion(logit, mask, is_lova=False)
        dice = model.dice(logit, mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())
        train_dice.append(dice)

        if 0:
            print('loss: {}'.format(loss.detach().cpu().item()))

        if DEBUG_mb:
            break

    # Validation:
    val_loss = do_valid(model, valid_loader, device=device, is_lova=False)

    if val_loss[2] > best_val_iou:
        best_val_iou = val_loss[2]
        torch.save(model.state_dict(), name_model_best_val_s1)  # to save the best val iou checkpoint

    log_str = "Epoch: %4d, Time: %.5f s  Train loss: %.5f  Train dice: %.5f  Val loss: %.5f  Val_dice: %.5f  Val_iou: %.5f " %(e,
                time.time() - e_t0, np.mean(train_loss), np.mean(train_dice), val_loss[0], val_loss[1], val_loss[2] )
    log.write(log_str)

    if DEBUG:
        break

log.write('best valid iou is {:.4f}\n\n'.format(best_val_iou))

""" ==================== Stage 2 ======================== """

STAGE_ID = 2

## outputs config
tag =  DEPTH_str + 'net_{}stage{}_fold_{}_'.format(UNetResNetstr[net_id], STAGE_ID, fold_id)  # net type && fold id

dir_log, dir_models, dir_subs = mkdir_outputs(STAGE_ID=STAGE_ID, subfolder_dir=subfolder_dir, debug=DEBUG)

# config log files
from logger import Logger
log_fn = dir_log + tag + time_id + '.txt'
print(log_fn)
log = Logger(log_fn)

fn_subs = dir_subs + tag + 'submission_' + time_id + '.csv'
log.write(fn_subs)

log.write('logging from script: ' + sys.argv[0] + '\n') # log the running file name

# config model files
name_model_best_val_s2 = dir_models + tag + 'best_val_iou_resnet34_' + time_id + '.ckpt'

#============= training =============#
# load best stage 1 model
model.load_state_dict(torch.load(name_model_best_val_s1))

optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=8, verbose=True)

best_val_iou = 0
count_early_stop = 0
for e in range(n_epochs_stage2):

    e_t0 = time.time()
    # Training:
    train_loss = []
    train_dice = []
    model.set_mode('train')
    for image, mask in (train_loader):
        image, mask = image.to(device), mask.to(device)

        if torch.cuda.device_count() > 1:
            logit = nn.parallel.data_parallel(model, image)
        else:
            logit = model(image)

        loss = model.criterion(logit, mask, is_lova=True)
        dice = model.dice(logit, mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())
        train_dice.append(dice)

        if 0:
            print('loss: {}'.format(loss.detach().cpu().item()))

        if DEBUG_mb:
            break

    # Validation:
    val_loss = do_valid(model, valid_loader, device=device, is_lova=True)
    scheduler.step(val_loss[2])

    if val_loss[2] > best_val_iou:
        best_val_iou = val_loss[2]
        count_early_stop = 0
        torch.save(model.state_dict(), name_model_best_val_s2)  # to save the best val iou checkpoint
    else:
        count_early_stop += 1


    log_str = "Epoch: %4d, Time: %.5f s  Train: %.5f, Train dice: %.5f, Val loss: %.5f, Val_dice: %.5f  Val_iou: %.5f " %(e,
                time.time() - e_t0, np.mean(train_loss), np.mean(train_dice), val_loss[0], val_loss[1], val_loss[2] )
    log.write(log_str)

    # Early Stop
    if count_early_stop >= 16: break

    if DEBUG:
        break

log.write('best valid iou is {:.4f}\n\n'.format(best_val_iou))

""" ==================== Test + Sub STAGE 2 ======================== """
model.load_state_dict(torch.load(name_model_best_val_s2))

model.set_mode('test')

# simple get models predicts on valid data
# preds, truth shape: [320 * 1 * 256 * 256]
def get_valid_preds_truths(model, valid_loader):
    preds, truth = [], []
    for image, mask in valid_loader:
        image = image.to(device)
        if torch.cuda.device_count() > 1:
            logit = nn.parallel.data_parallel(model, image)
        else:
            logit = model(image)
        preds.append(logit.cpu().detach().numpy())
        truth.append(mask.cpu().detach().numpy())
    preds = np.concatenate(preds)
    truth = np.concatenate(truth)
    return preds, truth

def Local_LB():
    preds, truth = get_valid_preds_truths(model, valid_loader)

    # unAUG
    preds, truth = de_pad2(preds, truth)
    preds = down_sample_array(preds.squeeze())
    truth = down_sample_array(truth.squeeze())
    truth = (truth > 0.5).astype(np.float32)
    precision, result, threshold = do_kaggle_metric(preds.squeeze(), truth.squeeze(), threshold=0)
    return precision.mean()

def npflip_array(x):
    out = np.zeros_like(x)
    for i in range(len(x)):
        out[i] = np.fliplr(x[i])
    return out

def get_test_preds(test_loader, model):
    preds = []
    for image, in test_loader:
        if torch.cuda.device_count() > 1:
            logit = nn.parallel.data_parallel(model, image.to(device))
        else:
            logit = model(image.to(device))
        logit = logit.cpu().detach().numpy()

        if TTA:
            if use_depth:
                image_t = torch.from_numpy(np.array([npflip_array(x) for x in image.numpy().squeeze()])).to(device)
            else:
                image_t = torch.from_numpy(np.expand_dims(np.array([np.fliplr(x) for x in image.numpy().squeeze()]), axis=1)).to(device)

            if torch.cuda.device_count() > 1:
                y_pred2 = nn.parallel.data_parallel(model, image_t)
            else:
                y_pred2 = model(image_t)

            y_pred2 = y_pred2.cpu().detach().numpy()
            logit += np.expand_dims(np.array([np.fliplr(x) for x in y_pred2.squeeze()]), axis=1)
            logit /= 2
        preds.append(logit)
    preds = np.concatenate(preds)
    return preds

log.write('*** local LB is {:.6f} ***'.format(Local_LB()))

if DEBUG is False:
    t0 = time.time()
    test_preds = get_test_preds(test_loader, model)
    # unAUG
    test_preds = de_pad(test_preds)
    test_preds = down_sample_array(test_preds.squeeze())

    binary_prediction = np.array([img for img in test_preds > 0.], dtype=int)

    # RLE encoding.
    all_masks = {idx:rle_encode(binary_prediction[i]) for i, idx in enumerate(test_ids)}
    submission = pd.DataFrame.from_dict(all_masks, orient='index')
    submission.index.names = ['id']
    submission.columns = ['rle_mask']
    submission.to_csv(fn_subs)
    log.write('Cost {:.4f} min to do test submission'.format((time.time()-t0)/60.))

""" ==================== Stage 3 ======================== """
STAGE_ID = 3
tag = DEPTH_str + 'net_{}stage{}_fold_{}_'.format(UNetResNetstr[net_id], STAGE_ID, fold_id)

# config outputs files
dir_log, dir_models, dir_subs = mkdir_outputs(STAGE_ID=STAGE_ID, subfolder_dir=subfolder_dir, debug=DEBUG)

# config log files
from logger import Logger
log_fn = dir_log + tag + time_id + '.txt'
print(log_fn)
log = Logger(log_fn)

fn_subs = dir_subs + tag + 'submission_' + time_id + '.csv'
log.write(fn_subs)

log.write('logging from script: ' + sys.argv[0] + '\n') # log the running file name

name_model_snap_stage3 =  dir_models + tag + 'snap_cyc_{:2d}_' + time_id + '.ckpt'

#============= training =============#
model.set_mode('train')
model.load_state_dict(torch.load(name_model_best_val_s2))

def proposed_lr(initial_lr, iteration, epoch_per_cycle):
    # proposed learning late function
    return initial_lr * ((cos(pi * iteration / epoch_per_cycle) + 1) / 2. * 0.9 + 0.1)  # range is (0.1 ~ 1) * 0.01

epochs_per_cycle = n_epochs_stage3 // cycles
initial_lr = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=initial_lr, momentum=0.9, weight_decay=0.0001)
for cyc in range(cycles):
    log.write("\n\nCycle: {:3d}".format(cyc))

    best_val_iou = 0
    best_model = None
    for e in range(epochs_per_cycle):
        lr = proposed_lr(initial_lr, e, epochs_per_cycle)
        # print(lr)
        optimizer.state_dict()["param_groups"][0]["lr"] = lr
        e_t0 = time.time()
        # Training:
        train_loss = []
        train_dice = []
        model.set_mode('train')
        for image, mask in (train_loader):
            image, mask = image.to(device), mask.to(device)

            if torch.cuda.device_count() > 1:
                logit = nn.parallel.data_parallel(model, image)
            else:
                logit = model(image)

            loss = model.criterion(logit, mask, is_lova=True)
            dice = model.dice(logit, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            train_dice.append(dice)

            if 0:
                print('loss: {}'.format(loss.detach().cpu().item()))

            if DEBUG_mb:
                break

        # Validation:
        val_loss = do_valid(model, valid_loader, device=device, is_lova=True)

        if val_loss[2] > best_val_iou:
            best_val_iou = val_loss[2]
            torch.save(model.state_dict(), name_model_snap_stage3.format(cyc))

        log_str = "Epoch: %4d, Time: %.5f s  Train loss: %.5f, Train dice: %.5f, Val loss: %.5f, Val_dice: %.5f  Val_iou: %.5f " %(e,
                    time.time() - e_t0, np.mean(train_loss), np.mean(train_dice), val_loss[0], val_loss[1], val_loss[2] )
        log.write(log_str)

        if DEBUG:
            break

""" ==================== Test + Sub stage 3 ======================== """

def do_ensumble_valid():
    model_list = [selected_net(depth=use_depth) for _ in range(cycles)]
    predict_final = np.zeros((len(valid_ids), 1, IMG_NET_SIZE, IMG_NET_SIZE))
    for i, model in enumerate(model_list):
        model.load_state_dict(torch.load(name_model_snap_stage3.format(i)))
        model.set_mode('test')
        model.cuda()

        valid_preds, valid_truths = get_valid_preds_truths(model, valid_loader)
        predict_final += valid_preds
    predict_final /= cycles

    # unAUG
    predict_final, valid_truths = de_pad2(predict_final, valid_truths)
    predict_final = down_sample_array(predict_final.squeeze())
    valid_truths = down_sample_array(valid_truths.squeeze())
    valid_truths = (valid_truths > 0.5).astype(np.float32)

    precision, result, threshold = do_kaggle_metric(predict_final, valid_truths, threshold=0.)
    return precision.mean()

def npflip_array(x):
    out = np.zeros_like(x)
    for i in range(len(x)):
        out[i] = np.fliplr(x[i])
    return out

def get_ensumble_preds():
    model_list = [selected_net(depth=use_depth) for _ in range(cycles)]
    predict_final = np.zeros((len(test_ids), 1, IMG_NET_SIZE, IMG_NET_SIZE))
    for i, model in enumerate(model_list):
        model.load_state_dict(torch.load(name_model_snap_stage3.format(i)))
        model.set_mode('test')
        model.cuda()

        test_predictions_stacked = get_test_preds(test_loader, model)
        predict_final += test_predictions_stacked
    predict_final /= cycles
    return predict_final

valid_ensumble_iou = do_ensumble_valid()
log.write('*** Ensumble Valid Iou is {:.6f} ***'.format(valid_ensumble_iou))

t0 = time.time()
predict_ensumble = get_ensumble_preds()

# unAUG
predict_ensumble = de_pad(predict_ensumble)
predict_ensumble = down_sample_array(predict_ensumble.squeeze())


threshold_best = 0.
binary_prediction = np.array([img for img in predict_ensumble > threshold_best], dtype=int)

# RLE encoding.
all_masks = {idx:rle_encode(binary_prediction[i]) for i, idx in enumerate(test_ids)}
submission = pd.DataFrame.from_dict(all_masks, orient='index')
submission.index.names = ['id']
submission.columns = ['rle_mask']
submission.to_csv(fn_subs)
log.write('Cost {:.4f} min to do test submission'.format((time.time()-t0)/60.))