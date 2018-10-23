from include import *


def dice_accuracy(prob, truth, threshold=0.5,  is_average=True):
    batch_size = prob.size(0)
    p = prob.detach().view(batch_size,-1)
    t = truth.detach().view(batch_size,-1)

    p = p>threshold
    t = t>0.5
    intersection = p & t
    union        = p | t
    dice = (intersection.float().sum(1)+EPS) / (union.float().sum(1)+EPS)

    if is_average:
        dice = dice.sum()/batch_size
        return dice
    else:
        return dice

def iou(img_true, img_pred):
    i = np.sum((img_true * img_pred) > 0)
    u = np.sum((img_true + img_pred) > 0)
    if u == 0:
        return u
    return i / u


def iou_metric(imgs_true, imgs_pred):
    num_images = len(imgs_true)
    scores = np.zeros(num_images)
    thresholds = np.arange(0.5, 1, 0.05)
    for i in range(num_images):
        if imgs_true[i].sum() == imgs_pred[i].sum() == 0:
            scores[i] = 1
        else:
            scores[i] = (thresholds <= iou(imgs_true[i], imgs_pred[i])).mean()

    return scores.mean()
def batch_iou_metric(label, pred):
    return iou_metric(label, pred>0.5)

def dice_metric(prob, truth, threshold=0.5,  is_average=True):
    batch_size = prob.size(0)
    p = prob.detach().view(batch_size,-1)
    t = truth.detach().contiguous().view(batch_size,-1)

    p = p>threshold
    t = t>0.5
    correct = ( p == t).float()
    accuracy = correct.sum(1)/p.size(1)

    if is_average:
        accuracy = accuracy.sum()/batch_size
        return accuracy
    else:
        return accuracy
def do_kaggle_metric(predict,truth, threshold=0.5):

    N = len(predict)
    predict = predict.reshape(N,-1)
    truth   = truth.reshape(N,-1)

    predict = predict>threshold
    truth   = truth>0.5
    intersection = truth & predict
    union        = truth | predict
    iou = intersection.sum(1)/(union.sum(1)+EPS)

    #-------------------------------------------
    result = []
    precision = []
    is_empty_truth   = (truth.sum(1)==0)
    is_empty_predict = (predict.sum(1)==0)

    threshold = np.array([0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95])
    for t in threshold:
        p = iou>=t

        tp  = (~is_empty_truth)  & (~is_empty_predict) & (iou> t)
        fp  = (~is_empty_truth)  & (~is_empty_predict) & (iou<=t)
        fn  = (~is_empty_truth)  & ( is_empty_predict)
        fp_empty = ( is_empty_truth)  & (~is_empty_predict)
        tn_empty = ( is_empty_truth)  & ( is_empty_predict)

        p = (tp + tn_empty) / (tp + tn_empty + fp + fp_empty + fn)

        result.append( np.column_stack((tp,fp,fn,tn_empty,fp_empty)) )
        precision.append(p)

    result = np.array(result).transpose(1,2,0)
    precision = np.column_stack(precision)
    precision = precision.mean(1)

    return precision, result, threshold
# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))
 
    print('\nsucess!')
