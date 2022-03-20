from PIL import Image
import numpy as np


train_lb20 = {
    7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
    19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
    26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18
}

train_lb16 = {
    7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
    19: 6, 20: 7, 21: 8, 23: 9, 24: 10, 25: 11,
    26: 12, 28: 13, 32: 14, 33: 15
}


def relabel(lbl, classes=20):
    train_lb = train_lb20 if classes == 20 else train_lb16

    lbl = np.asarray(lbl)
    lbl_copy = 255 * np.ones(lbl.shape, dtype=np.float32)
    for k, v in train_lb.items():
        lbl_copy[lbl == k] = v

    return Image.fromarray(lbl_copy)
