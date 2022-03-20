from relabel import relabel
import torch
import numpy as np
from PIL import Image
import random
import glob
import os


class LoaderGTA5:
    def __init__(self, root, split='images', batch_size=128, shuffle=True, img_transform=None, label_transform=None):
        # Assert that specified partition is valid
        assert split in ['images', 'train', 'test']

        self.root = root
        self.split = split

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.img_transform = img_transform
        self.label_transform = label_transform

        # If dataset is not partitioned, do it now
        if not os.path.exists(os.path.join(root, 'images.txt')):
            self._partition_dataset(root)

        # List input + target directories
        dir_img = os.path.join(root, 'images')
        dir_lb = os.path.join(root, 'labels')

        # List partition-specific images
        with open(os.path.join(root, split + '.txt'), 'r') as fp:
            p_images = [f[:-1] for f in fp.readlines()]

        # List all input files and corresponding target files
        self.files = []
        for im_name in p_images:
            im_path = os.path.join(dir_img, im_name)
            lb_path = os.path.join(dir_lb, im_name)
            self.files.append((im_path, lb_path))

        # Prepare indexing
        self.indices = np.arange(len(self.files))
        self.index = 0

    def __iter__(self):
        self.index = 0
        if self.shuffle:
            np.random.shuffle(self.indices)

        return self

    def next(self):
        if self.index >= len(self.indices):
            raise StopIteration

        image, target = [], []
        for i in self.indices[np.arange(self.index, np.minimum(len(self.files), self.index + self.batch_size))]:
            im_path, lb_path = self.files[i]

            img = self.img_transform(Image.open(im_path).convert('RGB'))
            label = self.label_transform(relabel(Image.open(lb_path).convert("P")))

            image.append(img)
            target.append(label)

        self.index += len(target)
        return torch.stack(image, 0), torch.stack(target, 0)

    def _partition_dataset(self, root):
        # List all images
        img_path = os.path.join(root, 'images')
        images = [im.split('/')[-1] + '\n' for im in glob.glob(img_path + '/*.png')]

        # Write list with all images
        with open(os.path.join(root, 'images.txt'), 'w') as f:
            f.writelines(images)

        # Separate 500 random images
        random.shuffle(images)
        train_images, test_images = images[:-500], images[-500:]

        # Write list with train images
        with open(os.path.join(root, 'train.txt'), 'w') as f:
            f.writelines(train_images)

        # Write list with test images
        with open(os.path.join(root, 'test.txt'), 'w') as f:
            f.writelines(test_images)
