import numpy as np
import cPickle as pkl
from PIL import Image
from glob import glob
import os


ignore_roi = False


def load_gtsrb(scale=False):
    data_target = pkl.load(open('/mnt/Storage/Datasets/domain_adaptation/synsig_gtsrb/gtsrb.pkl'))
    target_train = np.random.permutation(len(data_target['image']))
    data_t_im = data_target['image'][target_train[:31367], :, :, :]
    data_t_im_test = data_target['image'][target_train[31367:], :, :, :]
    data_t_label = np.cast[np.int64](data_target['label'][target_train[:31367]])
    data_t_label_test = np.cast[np.int64](data_target['label'][target_train[31367:]])
    data_t_im = data_t_im.transpose(0, 3, 1, 2).astype(np.float32)
    data_t_im_test = data_t_im_test.transpose(0, 3, 1, 2).astype(np.float32)
    return data_t_im, data_t_label, data_t_im_test, data_t_label_test


def build_gtsrb(path):
    # List all classes and files
    classes = [(int(cls), path + 'gtsrb/train/' + cls) for cls in next(os.walk(path + 'gtsrb/train/'))[1]]
    files = [glob(cl_f + '/*.ppm') for cl_i, cl_f in classes]

    # Prepare image and label lists
    images = []
    labels = []

    # Process images into 40x40 size
    for cl_i, cl_p in classes:
        # Load ground truth file
        with open(glob(cl_p + '/*.csv')[0], 'r') as fp:
            # Skip CSV header
            next(fp)

            # For each file in class GT
            for line in fp:
                f_name, _, _, l, u, r, b, _ = line.strip('\r\n').split(';')
                im = Image.open(cl_p + '/' + f_name)

                if not ignore_roi:
                    im = im.crop((int(l), int(u), int(r), int(b)))
                im = im.resize((40, 40))

                images.append(np.array(im))
                labels.append(cl_i)

    # Store dataset
    pkl.dump({
        'image': np.stack(images, axis=0),
        'label': np.array(labels),
    }, open(path + 'gtsrb.pkl', 'wb'), pkl.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    build_gtsrb('/mnt/Storage/Datasets/domain_adaptation/synsig_gtsrb/')
