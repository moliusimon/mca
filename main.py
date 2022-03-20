from __future__ import print_function
import argparse
import torch
import torch.nn as nn
from solver import Solver
import os

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MCD Implementation')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoint', metavar='N', help='source only or not')
parser.add_argument('--eval_only', action='store_true', default=False, help='evaluation only option')
parser.add_argument('--max_epoch', type=int, default=1000, metavar='N', help='how many epochs')
parser.add_argument('--resume_epoch', type=int, default=100, metavar='N', help='epoch to resume')
parser.add_argument('--save_epoch', type=int, default=10, metavar='N', help='when to restore the model')
parser.add_argument('--save_model', action='store_true', default=False, help='save_model or not')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--batch-size', type=int, default=128, metavar='N', help='input batch size for training (default: 128)')
parser.add_argument('--mca-weight', type=int, default=100, metavar='N', help='weight of the MCA loss (default: 100)')
parser.add_argument('--mca-projections', type=int, default=12, metavar='N', help='number of random projections for the MCA loss (default: 4)')

""" SVHN -> MNIST """
parser.add_argument('--source', type=str, default='svhn', metavar='N', help='source dataset')
parser.add_argument('--target', type=str, default='mnist', metavar='N', help='target dataset')

""" SYNSIG -> GTSRB """
# parser.add_argument('--source', type=str, default='synth', metavar='N', help='source dataset')
# parser.add_argument('--target', type=str, default='gtsrb', metavar='N', help='target dataset')

""" MNIST -> USPS """
# parser.add_argument('--source', type=str, default='mnist', metavar='N', help='source dataset')
# parser.add_argument('--target', type=str, default='usps', metavar='N', help='target dataset')

""" USPS -> MNIST """
# parser.add_argument('--source', type=str, default='usps', metavar='N', help='source dataset')
# parser.add_argument('--target', type=str, default='mnist', metavar='N', help='target dataset')

""" APPA-REAL """
# parser.add_argument('--source', type=str, default='appa', metavar='N', help='source dataset')
# parser.add_argument('--target', type=str, default='real', metavar='N', help='target dataset')

criterion = lambda: nn.CrossEntropyLoss().cuda()
# criterion = lambda: nn.MSELoss().cuda()

# Parse arguments
args = parser.parse_args()
args.cuda = torch.cuda.is_available()

# Set random seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)


def main():
    # lr = 0.0002
    solver = Solver(
        args, source=args.source, target=args.target,
        learning_rate=0.0005, batch_size=args.batch_size,
        mca_weight=args.mca_weight, mca_projections=args.mca_projections,
        optimizer='adam',
        checkpoint_dir=args.checkpoint_dir,
        save_epoch=args.save_epoch
    )

    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)

    if not os.path.exists('record'):
        os.mkdir('record')

    if args.eval_only:
        solver.test()
        return

    count = 0
    for t in xrange(args.max_epoch):
        count += solver.train(t, criterion=criterion())
        solver.test(criterion=criterion())


if __name__ == '__main__':
    main()
