from PIL import Image
import torchvision.transforms as transforms

from loaders import LoaderPaired, LoaderRaw, LoaderCityscapes, LoaderGTA5
from utils.transforms import ReLabel, ToLabel, Scale, RandomSizedCrop, RandomHorizontalFlip, RandomRotation

from svhn import load_svhn
from mnist import load_mnist
from usps import load_usps
from synth_traffic import load_syntraffic
from gtsrb import load_gtsrb
from appareal import load_appa, load_real


def return_dataset(data, scale=False):
    return {
        'svhn': load_svhn,
        'mnist': load_mnist,
        'usps': load_usps,
        'synth': load_syntraffic,
        'gtsrb': load_gtsrb,
        'appa': load_appa,
        'real': load_real,
    }[data](scale=scale)


def dataset_read(source, target, batch_size, scale=False):
    if source == 'gta' or source == 'synthia':
        n_class = 20 if source == 'gta' else 16

        img_augment = [
            RandomRotation(),
            RandomHorizontalFlip(),
            RandomSizedCrop()
        ]

        img_transform = [
            Scale((1024, 512), Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])
        ]

        label_transform = transforms.Compose([
            Scale((1024, 512), Image.NEAREST),
            ToLabel(),
            ReLabel(255, n_class - 1), # Last Class is "Void" or "Background" class
        ])

        if source == 'gta':
            ls_train = LoaderGTA5(
                '/data-local/data1-hdd/datasets/gta5',
                split='train', batch_size=batch_size, shuffle=True,
                img_transform=transforms.Compose(img_augment + img_transform),
                label_transform=label_transform
            )

            ls_test = LoaderGTA5(
                '/data-local/data1-hdd/datasets/gta5',
                split='test', batch_size=batch_size, shuffle=False,
                img_transform=transforms.Compose(img_transform),
                label_transform=label_transform
            )

        if source == 'synthia':
            raise NotImplementedError

        lt_train = LoaderCityscapes(
            '/data-local/data1-hdd/datasets/cityscapes',
            split='train', n_class=n_class, batch_size=batch_size, shuffle=True,
            img_transform=transforms.Compose(img_augment + img_transform),
            label_transform=label_transform
        )

        lt_test = LoaderCityscapes(
            '/data-local/data1-hdd/datasets/cityscapes',
            split='test', n_class=n_class, batch_size=batch_size, shuffle=True,
            img_transform=transforms.Compose(img_transform),
            label_transform=label_transform
        )

        # Return paired training and test sets
        return LoaderPaired(ls_train, lt_train), LoaderPaired(ls_test, lt_test)

    args = {'scale': scale}
    train_source, s_label_train, test_source, s_label_test = return_dataset(source, **args)
    train_target, t_label_train, test_target, t_label_test = return_dataset(target, **args)

    # Prepare train data for source & target
    S = {'imgs': train_source, 'labels': s_label_train}
    T = {'imgs': train_target, 'labels': t_label_train}

    # Prepare test data for source & target
    S_test = {'imgs': test_source, 'labels': s_label_test}
    T_test = {'imgs': test_target, 'labels': t_label_test}

    # Determine scale
    scale = 40 if (source == 'synth') else 28 if (source == 'usps' or target == 'usps') else 64 if (source == 'appa') else 32

    transform = [
        transforms.Scale(scale),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]

    if source == 'appa':
        transform = [
            RandomRotation(),
            RandomSizedCrop()
        ] + transform

    # Prepare input transform
    transform = transforms.Compose(transform)

    # Prepare paired training set
    dataset = LoaderPaired(
        LoaderRaw(S, batch_size, shuffle=True, img_transform=transform),
        LoaderRaw(T, batch_size, shuffle=True, img_transform=transform)
    )

    # Prepare paired test set
    dataset_test = LoaderPaired(
        LoaderRaw(S_test, batch_size, shuffle=True, img_transform=transform),
        LoaderRaw(T_test, batch_size, shuffle=True, img_transform=transform)
    )

    return dataset, dataset_test
