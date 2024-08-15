import numpy as np

def fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist

class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.epsilon = np.finfo(np.float32).eps

    def pixel_accuracy(self, hist):
        pa=np.diag(hist).sum() / hist.sum()
        return pa

    def mean_pixel_accuracy(self, hist):
        cpa = (np.diag(hist) + self.epsilon) / (hist.sum(axis=0) + self.epsilon)
        mpa = np.nanmean(cpa)
        return mpa

    def precision(self, hist):
        precision = (np.diag(hist) + self.epsilon) / (hist.sum(axis=0) + self.epsilon)
        precision = np.nanmean(precision)
        return precision

    def recall(self, hist):
        recall = (np.diag(hist) + self.epsilon) / (hist.sum(axis=1) + self.epsilon)
        recall = np.nanmean(recall)
        return recall

    def f1_score(self, hist):
        f1 = (np.diag(hist) + self.epsilon) * 2 / (hist.sum(axis=1) * 2 + hist.sum(axis=0) - np.diag(hist) + self.epsilon)
        f1 = np.nanmean(f1)
        return f1

    def mean_intersection_over_union(self, hist):
        iou = (np.diag(hist) + self.epsilon) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist) + self.epsilon)
        miou = np.nanmean(iou)
        return miou

    def frequency_weighted_intersection_over_union(self, hist):
        freq = hist.sum(axis=1) / hist.sum()
        iou = (np.diag(hist) + self.epsilon) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist) + self.epsilon)
        fwiou = (freq[freq > 0] * iou[freq > 0]).sum()
        return fwiou

    def cpa(self, hist):
        cpa = (np.diag(hist) + self.epsilon) / (hist.sum(axis=0) + self.epsilon)
        return cpa