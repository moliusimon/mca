from relabel import relabel
import torch
import numpy as np
import pickle as pkl
from PIL import Image
import random
import glob
import os


class LoaderAppaReal:
    def __init__(self, root, split='train', gender='male', batch_size=128, shuffle=True, img_transform=None):
        # Assert that specified partition and gender are valid
        assert split in ['train', 'test']
        assert gender in ['male', 'female']

        self.root = root
        self.split = split
        self.gender = gender

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.img_transform = img_transform

        # If dataset is not pre-processed, do it now
        data_path = os.path.join(root, 'dataset.pkl')
        if not os.path.exists(data_path):
            self._prepare_dataset(root)

        # Load samples for corresponding partition and filter by gender
        self.samples = [s for s in pkl.load(
            open(data_path, 'rb')
        )[split] if s['gender'] == gender]

        # Prepare indexing
        self.indices = np.arange(len(self.samples))
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
        for i in self.indices[np.arange(self.index, np.minimum(len(self.samples), self.index + self.batch_size))]:
            sample = self.samples[i]

            img = self.img_transform(sample['image'])
            label = np.array([sample['apparent'], sample['real']], dtype=np.float32)

            image.append(img)
            target.append(label)

        self.index += len(target)
        return torch.stack(image, 0), torch.stack(target, 0)

    @staticmethod
    def _prepare_dataset(root):
        train = LoaderAppaReal._build_partition(root, 'train')
        valid = LoaderAppaReal._build_partition(root, 'valid')
        test = LoaderAppaReal._build_partition(root, 'test')

        f_out = os.path.join(root, 'dataset.pkl')
        pkl.dump({
            'train': train,
            'test': valid + test,
        }, open(f_out, 'wb'), pkl.HIGHEST_PROTOCOL)

    @staticmethod
    def _build_partition(path, partition):
        files_path = os.path.join(path, partition)
        samples = {}

        # Open label file
        with open(os.path.join(path, 'gt_avg_' + partition + '.csv')) as fp:
            # Skip header line
            header = next(fp)

            # Read annotations
            for line in fp:
                fields = line[:-1].split(',')
                sid = int(fields[0].split('.')[0])
                f_path = os.path.join(files_path, fields[0] + '_face.jpg')

                samples[sid] = {
                    'image': np.array(Image.open(f_path).resize((256, 256))),
                    'apparent': float(fields[2]),
                    'real': float(fields[4])
                }

        # Open metadata file
        with open(os.path.join(path, 'allcategories_' + partition + '.csv')) as fp:
            # Skip header line
            header = next(fp)

            # Read annotations
            for line in fp:
                fields = line[:-1].split(',')
                sid = int(fields[0].split('.')[0])
                samples[sid]['gender'] = fields[1]
                samples[sid]['race'] = fields[2]

        # Convert labels to list and return
        return list(samples.values())
