from relabel import relabel
import torch
import numpy as np
from PIL import Image
import random
import glob
import os


class LoaderCityscapes:
    def __init__(self, root, split='train', n_class=20, batch_size=128, shuffle=True, img_transform=None, label_transform=None):
        # Assert that specified partition is valid
        assert split in ['train', 'val', 'test']

        self.root = root
        self.split = split

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.img_transform = img_transform or (lambda x: x)
        self.label_transform = label_transform or (lambda x: x)

        # List input + target file pairs
        dir_img = os.path.join(root, 'leftImg8bit/' + split)
        dir_lb = os.path.join(root, 'gtFine/' + split)

        # List all train files and corresponding test files
        self.files = []
        for group in os.walk(dir_img).next()[1]:
            tim_dir = os.path.join(dir_img, group)
            tlb_dir = os.path.join(dir_lb, group)
            for im_path in glob.glob(tim_dir + '/*.png'):
                signature = '_'.join(im_path.split('/')[-1].split('_')[:-1])
                lb_path = os.path.join(tlb_dir, signature + '_gtFine_labelIds.png')
                self.files.append((im_path, lb_path))

        if split == 'test':
            random.shuffle(self.files)
            self.files = self.files[:500]

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
