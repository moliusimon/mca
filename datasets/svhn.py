from scipy.io import loadmat
import numpy as np
from utils.utils import dense_to_one_hot

def load_svhn(scale=False):
    svhn_train = loadmat('/mnt/Storage/Datasets/domain_adaptation/svhn_mnist/train_32x32.mat')
    svhn_test = loadmat('/mnt/Storage/Datasets/domain_adaptation/svhn_mnist/test_32x32.mat')
    svhn_train_im = svhn_train['X']
    svhn_train_im = svhn_train_im.transpose(3, 2, 0, 1).astype(np.float32)
    svhn_label = np.cast[np.int64](dense_to_one_hot(svhn_train['y']))
    svhn_test_im = svhn_test['X']
    svhn_test_im = svhn_test_im.transpose(3, 2, 0, 1).astype(np.float32)
    svhn_label_test = np.cast[np.int64](dense_to_one_hot(svhn_test['y']))

    return svhn_train_im, svhn_label, svhn_test_im, svhn_label_test
