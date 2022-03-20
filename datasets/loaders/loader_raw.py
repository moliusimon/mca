import torch
import numpy as np
from PIL import Image


class LoaderRaw:
    def __init__(self, data, batch_size=128, shuffle=True, img_transform=None, label_transform=None):
        self.imgs, self.labels = data['imgs'], data['labels']

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.img_transform = img_transform or (lambda x: x)
        self.label_transform = label_transform or (lambda x: x)

        self.indices = np.arange(len(self.imgs))
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
        for i in self.indices[np.arange(self.index, np.minimum(len(self.imgs), self.index + self.batch_size))]:
            a, b = self.imgs[i], self.labels[i]

            a = np.uint8(np.asarray(a))
            a = np.vstack([a, a, a]) if a.shape[0] == 1 else a
            a = Image.fromarray(a.transpose((1, 2, 0)))

            image.append(self.img_transform(a))
            target.append(self.label_transform(b))

        self.index += len(target)
        return torch.stack(image, 0), torch.tensor(target)
