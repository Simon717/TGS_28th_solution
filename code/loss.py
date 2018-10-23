from include import *

#https://github.com/marvis/pytorch-yolo2/blob/master/FocalLoss.py
#https://github.com/unsky/focal-loss
class FocalLoss2d(nn.Module):

    def __init__(self, gamma=2, size_average=True):
        super(FocalLoss2d, self).__init__()
        self.gamma = gamma
        self.size_average = size_average


    def forward(self, logit, target, class_weight=None, type='softmax'):
        target = target.view(-1, 1).long()


        if type=='sigmoid':
            if class_weight is None:
                class_weight = [1]*2 #[0.5, 0.5]

            prob   = F.sigmoid(logit)
            prob   = prob.view(-1, 1)
            prob   = torch.cat((1-prob, prob), 1)
            select = torch.FloatTensor(len(prob), 2).zero_().cuda()
            select.scatter_(1, target, 1.)

        elif  type=='softmax':
            B,C,H,W = logit.size()
            if class_weight is None:
                class_weight =[1]*C #[1/C]*C

            logit   = logit.permute(0, 2, 3, 1).contiguous().view(-1, C)
            prob    = F.softmax(logit,1)
            select  = torch.FloatTensor(len(prob), C).zero_().cuda()
            select.scatter_(1, target, 1.)

        class_weight = torch.FloatTensor(class_weight).cuda().view(-1,1)
        class_weight = torch.gather(class_weight, 0, target)

        prob       = (prob*select).sum(1).view(-1,1)
        prob       = torch.clamp(prob,1e-8,1-1e-8)
        batch_loss = - class_weight *(torch.pow((1-prob), self.gamma))*prob.log()

        if self.size_average:
            loss = batch_loss.mean()
        else:
            # loss = torch.mean(torch.mean(batch_loss, 2), 2).suqeeze()
            # torch.mean(torch.mean(batch_loss.view(logit.shape), 2), 2).squeeze(1)
            loss = torch.mean(batch_loss.view(logit.shape[0], -1), 1)

        return loss

##------------

class RobustFocalLoss2d(nn.Module):
    #assume top 10% is outliers
    def __init__(self, gamma=2, size_average=True):
        super(RobustFocalLoss2d, self).__init__()
        self.gamma = gamma
        self.size_average = size_average


    def forward(self, logit, target, class_weight=None, type='softmax'):
        target = target.view(-1, 1).long()


        if type=='sigmoid':
            if class_weight is None:
                class_weight = [1]*2 #[0.5, 0.5]

            prob   = F.sigmoid(logit)
            prob   = prob.view(-1, 1)
            prob   = torch.cat((1-prob, prob), 1)
            select = torch.FloatTensor(len(prob), 2).zero_().cuda()
            select.scatter_(1, target, 1.)

        elif  type=='softmax':
            B,C,H,W = logit.size()
            if class_weight is None:
                class_weight =[1]*C #[1/C]*C

            logit   = logit.permute(0, 2, 3, 1).contiguous().view(-1, C)
            prob    = F.softmax(logit,1)
            select  = torch.FloatTensor(len(prob), C).zero_().cuda()
            select.scatter_(1, target, 1.)

        class_weight = torch.FloatTensor(class_weight).cuda().view(-1,1)
        class_weight = torch.gather(class_weight, 0, target)

        prob  = (prob*select).sum(1).view(-1,1)
        prob  = torch.clamp(prob,1e-8,1-1e-8)

        focus = torch.pow((1-prob), self.gamma)
        #focus = torch.where(focus < 2.0, focus, torch.zeros(prob.size()).cuda())
        focus = torch.clamp(focus,0,2)


        batch_loss = - class_weight *focus*prob.log()

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss

        return loss


##------------


##  http://geek.csdn.net/news/detail/126833
class PseudoBCELoss2d(nn.Module):
    def __init__(self):
        super(PseudoBCELoss2d, self).__init__()

    def forward(self, logit, truth):
        z = logit.view (-1)
        t = truth.view (-1)
        loss = z.clamp(min=0) - z*t + torch.log(1 + torch.exp(-z.abs()))
        loss = loss.sum()/len(t) #w.sum()
        return loss

from  torch.autograd import  Variable
def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1: # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard

def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.contiguous().view(-1)
    labels = labels.contiguous().view(-1)
    if ignore is None:
        return scores, labels
    valid = (labels != ignore)
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels

def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(np.isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n

def lovasz_hinge(logits, labels, per_image=True, ignore=None, is_average=True):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        if is_average == True:
            loss = mean(lovasz_hinge_flat(*flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore))
                    for log, lab in zip(logits, labels))
        else:
            loss = torch.zeros(logits.shape[0])
            # loss = [lovasz_hinge_flat(*flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore))
            #         for log, lab in zip(logits, labels)]
            for i, (log, lab) in enumerate(zip(logits, labels)):
                loss[i] = lovasz_hinge_flat(*flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore))
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss

def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * Variable(signs))
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    # loss = torch.dot(F.elu(errors_sorted)+1, Variable(grad))
    loss = torch.dot(F.relu(errors_sorted), Variable(grad))

    return loss
#
# #  https://github.com/bermanmaxim/jaccardSegment/blob/master/losses.py
# #  https://discuss.pytorch.org/t/solved-what-is-the-correct-way-to-implement-custom-loss-function/3568/4
# class CrossEntropyLoss2d(nn.Module):
#     def __init__(self, weight=None, size_average=True):
#         super(CrossEntropyLoss2d, self).__init__()
#         self.nll_loss = nn.NLLLoss2d(weight, size_average)
#
#     def forward(self, logits, targets):
#         return self.nll_loss(F.log_softmax(logits), targets)
#
# class BCELoss2d(nn.Module):
#     def __init__(self, weight=None, size_average=True):
#         super(BCELoss2d, self).__init__()
#         self.bce_loss = nn.BCELoss(weight, size_average)
#
#     def forward(self, logits, targets):
#         probs        = F.sigmoid(logits)
#         probs_flat   = probs.view (-1)
#         targets_flat = targets.view(-1)
#         return self.bce_loss(probs_flat, targets_flat)
#
#
# class SoftDiceLoss(nn.Module):
#     def __init__(self):  #weight=None, size_average=True):
#         super(SoftDiceLoss, self).__init__()
#
#
#     def forward(self, logits, targets):
#
#         probs = F.sigmoid(logits)
#         num = targets.size(0)
#         m1  = probs.view(num,-1)
#         m2  = targets.view(num,-1)
#         intersection = (m1 * m2)
#         score = 2. * (intersection.sum(1)+1) / (m1.sum(1) + m2.sum(1)+1)
#         score = 1- score.sum()/num
#         return score
#
#
#
# ##  http://geek.csdn.net/news/detail/126833
# class WeightedBCELoss2d(nn.Module):
#     def __init__(self):
#         super(WeightedBCELoss2d, self).__init__()
#
#     def forward(self, logits, labels, weights):
#         w = weights.view(-1)
#         z = logits.view (-1)
#         t = labels.view (-1)
#         loss = w*z.clamp(min=0) - w*z*t + w*torch.log(1 + torch.exp(-z.abs()))
#         loss = loss.sum()/(w.sum()+ 1e-12)
#         return loss
#
# class WeightedSoftDiceLoss(nn.Module):
#     def __init__(self):
#         super(WeightedSoftDiceLoss, self).__init__()
#
#     def forward(self, logits, labels, weights):
#         probs = F.sigmoid(logits)
#         num   = labels.size(0)
#         w     = (weights).view(num,-1)
#         w2    = w*w
#         m1    = (probs  ).view(num,-1)
#         m2    = (labels ).view(num,-1)
#         intersection = (m1 * m2)
#         score = 2. * ((w2*intersection).sum(1)+1) / ((w2*m1).sum(1) + (w2*m2).sum(1)+1)
#         score = 1 - score.sum()/num
#         return score
#
#

#

#
#
# def multi_loss(logits, labels):
#     #l = BCELoss2d()(logits, labels)
#
#
#     if 0:
#         l = BCELoss2d()(logits, labels) + SoftDiceLoss()(logits, labels)
#
#     #compute weights
#     else:
#         batch_size,C,H,W = labels.size()
#         weights = Variable(torch.tensor.torch.ones(labels.size())).cuda()
#
#         if 1: #use weights
#             kernel_size = 5
#             avg = F.avg_pool2d(labels,kernel_size=kernel_size,padding=kernel_size//2,stride=1)
#             boundary = avg.ge(0.01) * avg.le(0.99)
#             boundary = boundary.float()
#
#             w0 = weights.sum()
#             weights = weights + boundary*2
#             w1 = weights.sum()
#             weights = weights/w1*w0
#
#         l = WeightedBCELoss2d()(logits, labels, weights) + \
#             WeightedSoftDiceLoss()(logits, labels, weights)
#
#     return l
#
#
# #
# #
# #
# #
# #
# #
# #
# # class SoftCrossEntroyLoss(nn.Module):
# #     def __init__(self):
# #         super(SoftCrossEntroyLoss, self).__init__()
# #
# #     def forward(self, logits, soft_labels):
# #         #batch_size, num_classes =  logits.size()
# #         # soft_labels = labels.view(-1,num_classes)
# #         # logits      = logits.view(-1,num_classes)
# #
# #         logits = logits - logits.max()
# #         log_sum_exp = torch.log(torch.sum(torch.exp(logits), 1))
# #         loss = - (soft_labels*logits).sum(1) + log_sum_exp
# #         loss = loss.mean()
# #
# #         return loss
# #
# #
# #
# # # loss, accuracy -------------------------
# # def top_accuracy(probs, labels, top_k=(1,)):
# #     """Computes the precision@k for the specified values of k"""
# #
# #     probs  = probs.data
# #     labels = labels.data
# #
# #     max_k = max(top_k)
# #     batch_size = labels.size(0)
# #
# #     values, indices = probs.topk(max_k, dim=1, largest=True,  sorted=True)
# #     indices  = indices.t()
# #     corrects = indices.eq(labels.view(1, -1).expand_as(indices))
# #
# #     accuracy = []
# #     for k in top_k:
# #         # https://stackoverflow.com/questions/509211/explain-slice-notation
# #         # a[:end]      # items from the beginning through end-1
# #         c = corrects[:k].view(-1).float().sum(0, keepdim=True)
# #         accuracy.append(c.mul_(1. / batch_size))
# #     return accuracy
# #
# #
# # ## focal loss ## ---------------------------------------------------
# # class CrossEntroyLoss(nn.Module):
# #     def __init__(self):
# #         super(CrossEntroyLoss, self).__init__()
# #
# #     def forward(self, logits, labels):
# #         #batch_size, num_classes =  logits.size()
# #         # labels = labels.view(-1,1)
# #         # logits = logits.view(-1,num_classes)
# #
# #         max_logits  = logits.max()
# #         log_sum_exp = torch.log(torch.sum(torch.exp(logits-max_logits), 1))
# #         loss = log_sum_exp - logits.gather(dim=1, index=labels.view(-1,1)).view(-1) + max_logits
# #         loss = loss.mean()
# #
# #         return loss
# #
# # ## https://github.com/unsky/focal-loss
# # ## https://github.com/sciencefans/Focal-Loss
# # ## https://www.kaggle.com/c/carvana-image-masking-challenge/discussion/39951
# #
# # #  https://raberrytv.wordpress.com/2017/07/01/pytorch-kludges-to-ensure-numerical-stability/
# # #  https://github.com/pytorch/pytorch/issues/1620
# # class FocalLoss(nn.Module):
# #     def __init__(self,gamma = 2, alpha=1.2):
# #         super(FocalLoss, self).__init__()
# #         self.gamma = gamma
# #         self.alpha = alpha
# #
# #
# #     def forward(self, logits, labels):
# #         eps = 1e-7
# #
# #         # loss =  - np.power(1 - p, gamma) * np.log(p))
# #         probs = F.softmax(logits)
# #         probs = probs.gather(dim=1, index=labels.view(-1,1)).view(-1)
# #         probs = torch.clamp(probs, min=eps, max=1-eps)
# #
# #         loss = -torch.pow(1-probs, self.gamma) *torch.log(probs)
# #         loss = loss.mean()*self.alpha
# #
# #         return loss
# #
# #
# #
# #
# # # https://arxiv.org/pdf/1511.05042.pdf
# # class TalyorCrossEntroyLoss(nn.Module):
# #     def __init__(self):
# #         super(TalyorCrossEntroyLoss, self).__init__()
# #
# #     def forward(self, logits, labels):
# #         #batch_size, num_classes =  logits.size()
# #         # labels = labels.view(-1,1)
# #         # logits = logits.view(-1,num_classes)
# #
# #         talyor_exp = 1 + logits + logits**2
# #         loss = talyor_exp.gather(dim=1, index=labels.view(-1,1)).view(-1) /talyor_exp.sum(dim=1)
# #         loss = loss.mean()
# #
# #         return loss
# #
# # # check #################################################################
# # def run_check_focal_loss():
# #     batch_size  = 64
# #     num_classes = 15
# #
# #     logits = np.random.uniform(-2,2,size=(batch_size,num_classes))
# #     labels = np.random.choice(num_classes,size=(batch_size))
# #
# #     logits = Variable(torch.from_numpy(logits)).cuda()
# #     labels = Variable(torch.from_numpy(labels)).cuda()
# #
# #     focal_loss = FocalLoss(gamma = 2)
# #     loss = focal_loss(logits, labels)
# #     print (loss)
# #
# #
# # def run_check_soft_cross_entropy_loss():
# #     batch_size  = 64
# #     num_classes = 15
# #
# #     logits = np.random.uniform(-2,2,size=(batch_size,num_classes))
# #     soft_labels = np.random.uniform(-2,2,size=(batch_size,num_classes))
# #
# #     logits = Variable(torch.from_numpy(logits)).cuda()
# #     soft_labels = Variable(torch.from_numpy(soft_labels)).cuda()
# #     soft_labels = F.softmax(soft_labels,1)
# #
# #     soft_cross_entropy_loss = SoftCrossEntroyLoss()
# #     loss = soft_cross_entropy_loss(logits, soft_labels)
# #     print (loss)
#
# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))



    print('\nsucess!')