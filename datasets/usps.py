import numpy as np
import gzip
import cPickle


def load_usps(scale=False):
    f = gzip.open('/mnt/Storage/Datasets/domain_adaptation/usps/usps_28x28.pkl', 'rb')
    data_set = cPickle.load(f)
    f.close()
    img_train = data_set[0][0] * 255
    label_train = data_set[0][1]
    img_test = data_set[1][0] * 255
    label_test = np.cast[np.int64](data_set[1][1])
    inds = np.random.permutation(img_train.shape[0])
    img_train = img_train[inds]
    label_train = np.cast[np.int64](label_train[inds])
    img_train = img_train.reshape((img_train.shape[0], 1, 28, 28))
    img_test = img_test.reshape((img_test.shape[0], 1, 28, 28))
    return img_train, label_train, img_test, label_test
