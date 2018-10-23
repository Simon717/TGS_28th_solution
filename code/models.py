from common import *
import matplotlib.pyplot as plt
#  https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
#  resnet18 :  BasicBlock, [2, 2, 2, 2]
#  resnet34 :  BasicBlock, [3, 4, 6, 3]
#  resnet50 :  Bottleneck  [3, 4, 6, 3]
#

# https://medium.com/neuromation-io-blog/deepglobe-challenge-three-papers-from-neuromation-accepted-fe09a1a7fa53
# https://github.com/ternaus/TernausNetV2
# https://github.com/neptune-ml/open-solution-salt-detection
# https://github.com/lyakaap/Kaggle-Carvana-3rd-Place-Solution


##############################################################3
#  https://github.com/neptune-ml/open-solution-salt-detection/blob/master/src/unet_models.py
#  https://pytorch.org/docs/stable/torchvision/models.html

import torchvision


from metric import batch_iou_metric, dice_metric
from config import *

class ConvBn2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1)):
        super(ConvBn2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        # self.bn = SynchronizedBatchNorm2d(out_channels)



    def forward(self, z):
        x = self.conv(z)
        x = self.bn(x)
        return x

class SCSEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SCSEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.channel_excitation = nn.Sequential(nn.Linear(channel, int(channel//reduction)),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(int(channel//reduction), channel),
                                                nn.Sigmoid())

        self.spatial_se = nn.Sequential(nn.Conv2d(channel, 1, kernel_size=1,
                                                  stride=1, padding=0, bias=False),
                                        nn.Sigmoid())

    def forward(self, x):
        bahs, chs, _, _ = x.size()

        # Returns a new tensor with the same data as the self tensor but of a different size.
        chn_se = self.avg_pool(x).view(bahs, chs)
        chn_se = self.channel_excitation(chn_se).view(bahs, chs, 1, 1)
        chn_se = torch.mul(x, chn_se)

        spa_se = self.spatial_se(x)
        spa_se = torch.mul(x, spa_se)
        return torch.add(chn_se, 1, spa_se)

class Decoder(nn.Module):
    def __init__(self, in_channels, channels, out_channels , bn=True):
        super(Decoder, self).__init__()
        if bn:
            self.conv1 = ConvBn2d(in_channels, channels, kernel_size=3, padding=1)
            self.conv2 = ConvBn2d(channels, out_channels, kernel_size=3, padding=1)
            self.se = SCSEBlock(out_channels)
        else:
            self.conv1 =  nn.Conv2d(in_channels,  channels, kernel_size=3, padding=1)
            self.conv2 =  nn.Conv2d(channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x, e=None):
        x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        if e is not None:
            x = torch.cat([x, e], 1)
        x = F.relu(self.conv1(x),inplace=True)
        x = F.relu(self.conv2(x),inplace=True)
        x = self.se(x)
        return x

class Fuse(nn.Module):
    def __init__(self, in_channels, channels, out_channels):
        super(Fuse, self).__init__()
        self.conv1 = ConvBn2d(in_channels, channels, kernel_size=3, padding=1)
        self.conv2 = ConvBn2d(channels, out_channels, kernel_size=3, padding=1)
        self.se = SCSEBlock(out_channels)

    def forward(self, x):
        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        x = self.se(x)
        return x


class Decoder_OLD(nn.Module):
    def __init__(self, in_channels, channels, out_channels , bn=True, up=True):
        super(Decoder_OLD, self).__init__()
        self.up = up
        if bn:
            self.conv1 = ConvBn2d(in_channels, channels, kernel_size=3, padding=1)
            self.conv2 = ConvBn2d(channels, out_channels, kernel_size=3, padding=1)
            self.se = SCSEBlock(out_channels)
        else:
            self.conv1 =  nn.Conv2d(in_channels,  channels, kernel_size=3, padding=1)
            self.conv2 =  nn.Conv2d(channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x, e=None):
        if e is not None:
            x = torch.cat([x, e], 1)
        if self.up == True:
            x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)#False
        # x = F.conv_transpose2d(x,)
        x = F.relu(self.conv1(x),inplace=True)
        x = F.relu(self.conv2(x),inplace=True)
        x = self.se(x)
        return x

#
# resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
# resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth'
# resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
# resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'
# resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth'

def add_depth_channels(image_tensor):
    out =  image_tensor.repeat(3,1,1)
    # out[0] = image_tensor.squeeze()
    for row, const in enumerate(np.linspace(0, 1, 101)):
        out[1, 13+row, 13:-14] = const
    out[2, 13:-14, 13:-14] = out[0, 13:-14, 13:-14] * out[1, 13:-14, 13:-14]
    return out

class Hy_UNetResNet34(nn.Module):
    # PyTorch U-Net model using ResNet(34, 50 , 101 or 152) encoder.

    def load_pretrain(self, pretrain_file):
        self.resnet.load_state_dict(torch.load(pretrain_file, map_location=lambda storage, loc: storage))
        # self.encoder.load_state_dict(torch.load(pretrain_file, map_location=lambda storage, loc: storage))

    def __init__(self, depth=False):
        super().__init__()
        self.depth  = depth
        self.resnet = torchvision.models.resnet34()

        self.conv1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
        )# 64
        self.encoder2 = self.resnet.layer1  # 64
        self.encoder3 = self.resnet.layer2  #128
        self.encoder4 = self.resnet.layer3  #256
        self.encoder5 = self.resnet.layer4  #512

        self.center = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.decoder5 = Decoder(256+512, 512, 64)
        self.decoder4 = Decoder(64+256,  256, 64)
        self.decoder3 = Decoder(64+128,  128, 64)
        self.decoder2 = Decoder(64+ 64,  64,  64)
        self.decoder1 = Decoder(64 ,     32,  64)


        self.logit    = nn.Sequential(
            nn.Conv2d(320, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,  1, kernel_size=1, padding=0),
        )



    def forward(self, x):
        #batch_size,C,H,W = x.shape
        if self.depth:
            x_ex = [add_depth_channels(a) for a in x]
            x = torch.cat(x_ex)
            x = x.view(-1, 3, 128, 128).cuda()
        else:
            mean=[0.485, 0.456, 0.406]
            std =[0.229, 0.224, 0.225]
            x = torch.cat([
                (x-mean[2])/std[2],
                (x-mean[1])/std[1],
                (x-mean[0])/std[0],
            ],1)
        # with torch.no_grad():
        x = self.conv1(x)        #; print('x', x.size())
        e2 = self.encoder2(x)    #; print('e2',e2.size())
        e3 = self.encoder3(e2)   #; print('e3',e3.size())
        e4 = self.encoder4(e3)   #; print('e4',e4.size())
        e5 = self.encoder5(e4)   #; print('e5',e5.size())

        f = self.center(e5)          #; print('c', f.size())
        d5 = self.decoder5(f, e5)    #; print('d5',d5.size())
        d4 = self.decoder4(d5, e4)   #; print('d4',d4.size())
        d3 = self.decoder3(d4, e3)   #; print('d3',d3.size())
        d2 = self.decoder2(d3, e2)   #; print('d2',d2.size())
        d1 = self.decoder1(d2)       #; print('d1',d1.size())

        f = torch.cat((d1,
                      F.upsample(d2, scale_factor=2,  mode='bilinear', align_corners=False),
                      F.upsample(d3, scale_factor=4,  mode='bilinear', align_corners=False),
                      F.upsample(d4, scale_factor=8,  mode='bilinear', align_corners=False),
                      F.upsample(d5, scale_factor=16, mode='bilinear', align_corners=False)),1)
        f = F.dropout(f, p=0.4)
        logit = self.logit(f)                     #; pint('logit',logit.size())
        return logit


    ##-----------------------------------------------------------------


    def criterion(self, logit, truth ):

        #loss = PseudoBCELoss2d()(logit, truth)
        loss = FocalLoss2d()(logit, truth, type='sigmoid')
        # loss = RobustFocalLoss2d()(logit, truth, type='sigmoid')
        return loss

    def dice(self, logit, truth, threshold=0.5 ):
        prob = F.sigmoid(logit)
        dice = dice_metric(prob, truth, threshold=threshold, is_average=True)
        return dice


    def model_iou_metric(self, logit, truth, threshold=0.5 ):
        # prob = F.sigmoid(logit)
        # dice = dice_accuracy(prob, truth, threshold=threshold, is_average=True)
        prob = F.sigmoid(logit)
        prob = prob.cpu().detach().numpy()
        iou = batch_iou_metric(truth, prob)
        return iou

    def model_iou_metric_lova(self, logit, truth):
        return iou_metric(truth, logit > 0.)

    def lova_loss(self, logit, truth):
        return lovasz_hinge(logit, truth)



    def set_mode(self, mode ):
        self.mode = mode
        if mode in ['eval', 'valid', 'test']:
            self.eval()
        elif mode in ['train']:
            self.train()
        else:
            raise NotImplementedError

from resnet_base import ResNet, BasicBlock

class Hy_EN_UNetResNet34(nn.Module):
    # PyTorch U-Net model using ResNet(34, 50 , 101 or 152) encoder.

    def load_pretrain(self, pretrain_file):
        pretrained_dict = torch.load(pretrain_file, map_location=lambda storage, loc: storage)
        model_dict = self.resnet.state_dict()
        pretrained_dict = {k: v for k,v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.resnet.load_state_dict(model_dict)

        # self.resnet.load_state_dict(torch.load(pretrain_file, map_location=lambda storage, loc: storage))
        # self.encoder.load_state_dict(torch.load(pretrain_file, map_location=lambda storage, loc: storage))

    def __init__(self, depth=False):
        super().__init__()
        self.depth  = depth
        # self.resnet = torchvision.models.resnet34()
        self.resnet = ResNet(BasicBlock, [3, 4, 6, 3])

        self.conv1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
        )# 64
        self.encoder2 = self.resnet.layer1  # 64
        self.encoder3 = self.resnet.layer2  #128
        self.encoder4 = self.resnet.layer3  #256
        self.encoder5 = self.resnet.layer4  #512

        self.center = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.decoder5 = Decoder(256+512, 512, 64)
        self.decoder4 = Decoder(64+256,  256, 64)
        self.decoder3 = Decoder(64+128,  128, 64)
        self.decoder2 = Decoder(64+ 64,  64,  64)
        self.decoder1 = Decoder(64 ,     32,  64)


        self.logit    = nn.Sequential(
            nn.Conv2d(320, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,  1, kernel_size=1, padding=0),
        )



    def forward(self, x):
        if self.depth == False:             # 如果不用深度信息 输入的 shape = [1, 256, 256]
            mean = [0.485, 0.456, 0.406]    # 使用减均值 除方差 做预处理
            std = [0.229, 0.224, 0.225]     # 如果使用深度信息 不需要做预处理
            x = torch.cat([
                (x - mean[2]) / std[2],
                (x - mean[1]) / std[1],
                (x - mean[0]) / std[0],
            ], 1)

        x = self.conv1(x)            #; print('x', x.size())   # x  torch.Size([8, 64, 64, 64])
        e2 = self.encoder2(x)        #; print('e2',e2.size())  # e2 torch.Size([8, 64, 64, 64])
        e3 = self.encoder3(e2)       #; print('e3',e3.size())  # e3 torch.Size([8, 128, 32, 32])
        e4 = self.encoder4(e3)       #; print('e4',e4.size())  # e4 torch.Size([8, 256, 16, 16])
        e5 = self.encoder5(e4)       #; print('e5',e5.size())  # e5 torch.Size([8, 512, 8, 8])

        f = self.center(e5)          #; print('c', f.size())   # c torch.Size([8, 256, 4, 4])

        d5 = self.decoder5(f, e5)    #; print('d5',d5.size())  # d5 torch.Size([8, 64, 8, 8])   # decoder 设计为 先对自己上采样 然后再cat 卷积
        d4 = self.decoder4(d5, e4)   #; print('d4',d4.size())  # d4 torch.Size([8, 64, 16, 16])
        d3 = self.decoder3(d4, e3)   #; print('d3',d3.size())  # d3 torch.Size([8, 64, 32, 32])
        d2 = self.decoder2(d3, e2)   #; print('d2',d2.size())  # d2 torch.Size([8, 64, 64, 64])
        d1 = self.decoder1(d2)       #; print('d1',d1.size())  # d1 torch.Size([8, 64, 128, 128])

        f = torch.cat((d1,
                      F.upsample(d2, scale_factor=2,  mode='bilinear', align_corners=False),
                      F.upsample(d3, scale_factor=4,  mode='bilinear', align_corners=False),
                      F.upsample(d4, scale_factor=8,  mode='bilinear', align_corners=False),
                      F.upsample(d5, scale_factor=16, mode='bilinear', align_corners=False)),1)

        f = F.dropout(f, p=0.4)
        logit = self.logit(f)                     #; pint('logit',logit.size())
        return logit


    ##-----------------------------------------------------------------


    def criterion(self, logit, truth, is_lova=False):
        if is_lova:
            loss = lovasz_hinge(logit, truth)
        else:
            loss = FocalLoss2d()(logit, truth, type='sigmoid')

        return loss

    def dice(self, logit, truth, threshold=0.5 ):
        prob = F.sigmoid(logit)
        dice = dice_metric(prob, truth, threshold=threshold, is_average=True)
        return dice

    #
    # def model_iou_metric(self, logit, truth, threshold=0.5 ):
    #     # prob = F.sigmoid(logit)
    #     # dice = dice_accuracy(prob, truth, threshold=threshold, is_average=True)
    #     prob = F.sigmoid(logit)
    #     prob = prob.cpu().detach().numpy()
    #     iou = batch_iou_metric(truth, prob)
    #     return iou
    #
    # def model_iou_metric_lova(self, logit, truth):
    #     return iou_metric(truth, logit > 0.)
    #
    # def lova_loss(self, logit, truth):
    #     return lovasz_hinge(logit, truth)
    #


    def set_mode(self, mode ):
        self.mode = mode
        if mode in ['eval', 'valid', 'test']:
            self.eval()
        elif mode in ['train']:
            self.train()
        else:
            raise NotImplementedError

### run ##############################################################################

def run_check_net():

    batch_size = 8
    C,H,W = 1, 128, 128

    input = np.random.uniform(0,1, (batch_size,C,H,W)).astype(np.float32)
    truth = np.random.choice (2,   (batch_size,C,H,W)).astype(np.float32)
    truth_image = np.random.choice(2, (batch_size, )).astype(np.float32)


    #------------
    input = torch.from_numpy(input).float().cuda()
    truth = torch.from_numpy(truth).float().cuda()
    truth_image = torch.from_numpy(truth_image).float().cuda()


    #---
    net = Hy_EN_UNetResNet34().cuda()
    # net = Supervise_HENG().cuda()
    net.set_mode('train')
    # print(net)
    # exit(0)

    net.load_pretrain('/home/simon/code/20180921/models/resnet34-333f7ec4.pth')

    # logit = net(input)
    # loss  = net.criterion(logit, truth)

    logit, logit_pixel, logit_image = net(input)
    loss0, loss1, loss2 = net.criterion(logit, logit_pixel, logit_image, truth, truth_image)
    loss = loss0 + loss1 +  loss2
    # dice  = net.iou_metric(logit, truth)

    print('loss : %0.8f'%loss.item())
    # print('dice : %0.8f'%dice.item())
    print('')


    # dummy sgd to see if it can converge ...
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                      lr=0.1, momentum=0.9, weight_decay=0.0001)

    #optimizer = optim.Adam(net.parameters(), lr=0.001)


    i=0
    optimizer.zero_grad()
    while i<=500:

        logit = net(input)
        # loss  = net.criterion(logit, truth)
        # dice  = net.iou_metric(logit, truth)
        logit, logit_pixel, logit_image = net(input)
        loss0, loss1, loss2 = net.criterion(logit, logit_pixel, logit_image, truth, truth_image)
        loss = loss0 + loss1 + loss2

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i%20==0:
            print('[%05d] loss, dice: %.4f' %(i, loss.item()))
        i = i+1


########################################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_check_net()

    print( 'sucessful!')