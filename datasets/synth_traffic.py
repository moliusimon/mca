import numpy as np
import cPickle as pkl
from PIL import Image


def load_syntraffic(scale=False):
    data_source = pkl.load(open('/mnt/Storage/Datasets/domain_adaptation/synsig_gtsrb/synsig.pkl'))
    source_train = np.random.permutation(len(data_source['image']))
    data_s_im = data_source['image'][source_train[:len(data_source['image'])], :, :, :]
    data_s_im_test = data_source['image'][source_train[len(data_source['image']) - 2000:], :, :, :]
    data_s_label = np.cast[np.int64](data_source['label'][source_train[:len(data_source['image'])]])
    data_s_label_test = np.cast[np.int64](data_source['label'][source_train[len(data_source['image']) - 2000:]])
    data_s_im = data_s_im.transpose(0, 3, 1, 2).astype(np.float32)
    data_s_im_test = data_s_im_test.transpose(0, 3, 1, 2).astype(np.float32)
    return data_s_im, data_s_label, data_s_im_test, data_s_label_test


def build_syntraffic(path):
    images, labels = [], []

    # Read ground truth file
    with open(path + 'train.csv', 'r') as fp:
        for line in fp:
            f_name, label = line.split(' ')[:2]
            im = Image.open(path + f_name).resize((40, 40))
            images.append(np.array(im))
            labels.append(int(label))

    # Store dataset
    pkl.dump({
        'image': np.stack(images, axis=0),
        'label': np.array(labels),
    }, open(path + 'synsig.pkl', 'wb'), pkl.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    build_syntraffic('/data-local/data1-hdd/datasets/synsig/')
