import numpy as np
import cPickle as pkl
from PIL import Image
import csv


def load_domain(gender=0):
    d_target = pkl.load(open('/data-local/data1-hdd/datasets/appa-real/appareal.pkl'))
    d_train, d_test = d_target[0], d_target[1]

    # Prepare response arrays
    im_train, im_test = [], []
    lb_train, lb_test = [], []

    # Prepare first domain (male) train data
    for k, v in d_train.items():
        if v['gender'] == gender:
            im_train.append(v['image'])
            lb_train.append(v['age_apparent'])

    # Prepare first domain (male) test data
    for k, v in d_test.items():
        if v['gender'] == gender:
            im_test.append(v['image'])
            lb_test.append(v['age_apparent'])

    # Prepare original train/test sets
    im_train = np.stack(im_train, 0)
    lb_train = np.cast[np.float32](np.stack(lb_train, 0).reshape((-1, 1))) / 50 - 1
    im_test = np.stack(im_test, 0)
    lb_test = np.cast[np.float32](np.stack(lb_test, 0).reshape((-1, 1))) / 50 - 1

    # Augment data with mirror images
    im_train = np.concatenate((im_train, im_train[..., ::-1]), axis=0)
    lb_train = np.concatenate((lb_train, lb_train), axis=0)
    im_test = np.concatenate((im_test, im_test[..., ::-1]), axis=0)
    lb_test = np.concatenate((lb_test, lb_test), axis=0)

    # Return train and test data
    return im_train, lb_train, im_test, lb_test


def load_appa(scale=False):
    return load_domain(gender=0)


def load_real(scale=False):
    return load_domain(gender=1)


def build_partition(path, part='train'):
    data = {}

    # Load age data
    with open(path + 'gt_avg_' + part + '.csv') as fp:
        # Open file and skip header
        csf = csv.reader(fp)
        next(csf)

        # Read all lines
        for row in csf:
            # Capture metadata
            ind = int(row[0].split('.')[0])
            data[ind] = {
                'age_apparent': float(row[2]),
                'age_real': int(row[4]),
            }

            # Prepare image
            img = Image.open(open(path + part + '/' + row[0] + '_face.jpg', 'rb'))
            data[ind]['image'] = np.array(img.resize((64, 64))).transpose((2, 0, 1))

    # Load related metadata
    with open(path + 'allcategories_' + part + '.csv') as fp:
        # Open file and skip header
        csf = csv.reader(fp)
        next(csf)

        # Read all lines
        for row in csf:
            ind = int(row[0].split('.')[0])
            data[ind].update({
                'gender': 0 if row[1] == 'male' else 1,
                'ethnicity': row[2],
            })

    return data


def build_appareal(path):
    # Prepare partitions
    dataset = (
        build_partition(path, 'train'),
        build_partition(path, 'test'),
    )

    # Write processed dataset to file
    pkl.dump(dataset, open(path + 'appareal.pkl', 'wb'), pkl.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    build_appareal('/data-local/data1-hdd/datasets/appa-real/')
