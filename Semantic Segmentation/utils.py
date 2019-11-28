import numpy as np
import torch

class RemoveContour(object):
    """remove contour of given PIL Image.
    Args:
        img (PIL Image): Image

    Returns:
        PIL Image: image
    """

    def __init__(self, remove=True):
        self.remove = remove
        
    def __call__(self, img):
        img_np = np.array(img)

        # to remove the contours, and set them as background
        img_np[img_np == 255] = 0

        return img_np


def unnormalize(imgs):
    """unnormalize given set of Images.
    """
    shot = np.zeros((len(imgs),3,imgs.size()[2],imgs.size()[3]))
    shot = torch.from_numpy(shot)
    for i,data in enumerate(imgs.cpu()):
        for j, (t, m, s) in enumerate(zip(data, [0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])):        
            shot[i,j,:,:] = t.mul_(s).add_(m)  
    return shot

def get_confusion_matrix(pred, target, num_classes):
    """
    a quick way to calculate confusion matrix (adopted from https://github.com/kevinzakka/pytorch-goodies/blob/master/metrics.py)
    """
    mask = (target >= 0) & (target < num_classes)
    cm = np.bincount(
        num_classes * target[mask].astype(int) +
        pred[mask], minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return cm

def evaluate(cm):
    """
    function to calculate iou and dice score
    Args:
        cm: confusion matrix
    Returns
        iou, dice
    """
    tp = np.diag(cm) 
    tp_fp = cm.sum(axis=1) 
    tp_fn = cm.sum(axis=0)
    
    dice_score = (2 * tp) / (tp_fp + tp_fn)
    mean_dice = np.nanmean(dice_score)    # we use nanmean rather than mean inorder to ignore any nan values in the dice score
    
    iou = tp / (tp_fp + tp_fn - tp)
    mean_iou = np.nanmean(iou) 
    
    return mean_iou, mean_dice

def get_pascal_label_colormap():
        """Load the mapping that associates pascal classes with label colors
        Returns:
            array of colormap
        """
        return np.asarray(
            [
                [0, 0, 0],
                [128, 0, 0],
                [0, 128, 0],
                [128, 128, 0],
                [0, 0, 128],
                [128, 0, 128],
                [0, 128, 128],
                [128, 128, 128],
                [64, 0, 0],
                [192, 0, 0],
                [64, 128, 0],
                [192, 128, 0],
                [64, 0, 128],
                [192, 0, 128],
                [64, 128, 128],
                [192, 128, 128],
                [0, 64, 0],
                [128, 64, 0],
                [0, 192, 0],
                [128, 192, 0],
                [0, 64, 128],
            ]
        )
    
def get_label_colormap(mask):
        """Adds colormap to the mask
        Args:
            mask: a 2D array of integer values storing segmentation labels
        Returns:
            2D array of resulting colored label mask.
        """
        label_colours = get_pascal_label_colormap()
        r = mask.copy()
        g = mask.copy()
        b = mask.copy()
        for ll in range(0, 21):
            r[mask == ll] = label_colours[ll, 0]
            g[mask == ll] = label_colours[ll, 1]
            b[mask == ll] = label_colours[ll, 2]
        rgb = np.zeros((mask.shape[0], mask.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb    