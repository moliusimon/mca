from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from model import build_generator, build_classifier
from datasets.dataset_read import dataset_read
from mca import MCALoss, pseudo_labelled_cross_entropy_loss
import numpy as np


# Training settings
class Solver(object):
    def __init__(self, args, batch_size=64, source='svhn', target='mnist',
                 optimizer='adam', learning_rate=0.0002, mca_weight=100, mca_projections=4,
                 interval=100, checkpoint_dir=None, save_epoch=10):
        self.batch_size = batch_size
        self.source = source
        self.target = target
        self.interval = interval
        self.checkpoint_dir = checkpoint_dir
        self.save_epoch = save_epoch
        self.scale = (self.source == 'svhn')

        print('dataset loading')
        self.datasets, self.dataset_test = dataset_read(source, target, self.batch_size, scale=self.scale)
        print('load finished!')

        self.G = build_generator(source=source, target=target)
        self.C1 = build_classifier(source=source, target=target)
        self.MCA = MCALoss(batch_size, projections=mca_projections, mca_weight=mca_weight)
        if args.eval_only:
            self.G.torch.load(
                '%s/%s_to_%s_model_epoch%s_G.pt' % (self.checkpoint_dir, self.source, self.target, args.resume_epoch))
            self.G.torch.load(
                '%s/%s_to_%s_model_epoch%s_G.pt' % (
                    self.checkpoint_dir, self.source, self.target, self.checkpoint_dir, args.resume_epoch))
            self.G.torch.load(
                '%s/%s_to_%s_model_epoch%s_G.pt' % (self.checkpoint_dir, self.source, self.target, args.resume_epoch))

        self.G.cuda()
        self.C1.cuda()
        self.set_optimizer(which_opt=optimizer, lr=learning_rate)

    def set_optimizer(self, which_opt='momentum', lr=0.001, momentum=0.9):
        args = {'lr': lr, 'weight_decay': 0.0005}
        if which_opt == 'momentum':
            args['momentum'] = momentum

        opt = {'momentum': optim.SGD, 'adam': optim.Adam}[which_opt]
        self.opt_g = opt(self.G.parameters(), **args)
        self.opt_c1 = opt(self.C1.parameters(), **args)

    def reset_grad(self):
        self.opt_g.zero_grad()
        self.opt_c1.zero_grad()

    def ent(self, output):
        return - torch.mean(output * torch.log(output + 1e-6))

    def discrepancy(self, out1, out2):
        return torch.mean(torch.abs(F.softmax(out1) - F.softmax(out2)))

    def train(self, epoch, criterion=nn.CrossEntropyLoss().cuda()):
        self.G.train()
        self.C1.train()
        torch.cuda.manual_seed(1)

        n_batch = 500
        for batch_idx, data in enumerate(self.datasets):
            img_s = data['S'].cuda()
            img_t = data['T'].cuda()
            label_s = data['S_label'].cuda()
            if img_s.size()[0] < self.batch_size or img_t.size()[0] < self.batch_size:
                break

            self.reset_grad()
            feat_s, feat_t = self.G(img_s), self.G(img_t)
            loss_mca = self.MCA(feat_s, feat_t, epoch * n_batch + batch_idx)
            output_s1 = self.C1(feat_s)

            # loss_s1 = pseudo_labelled_cross_entropy_loss(output_s1, self.C1(feat_t), label_s)
            loss_s1 = criterion(output_s1, label_s)
            loss_s = loss_s1 + loss_mca

            loss_s.backward()
            self.opt_g.step()
            self.opt_c1.step()
            self.reset_grad()

            if batch_idx > n_batch:
                return batch_idx

            if batch_idx % self.interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx, n_batch,
                    100 * batch_idx / n_batch, loss_s1.item()))

        return batch_idx

    def test(self, criterion=nn.CrossEntropyLoss().cuda()):
        self.G.eval()
        self.C1.eval()

        test_loss = 0
        correct = 0
        n_batch = 0
        size = 0

        intersect = 0
        union = 0

        for batch_idx, (img, label) in enumerate(self.dataset_test.data_loader_B):
            img, label = img.cuda(), label.cuda()
            output = self.C1(self.G(img))

            test_loss += criterion(output, label).item()
            pred = output.data.max(1)[1]
            correct += pred.eq(label.data).sum().cpu()

            n_batch += 1
            size += np.prod(label.shape)

        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            test_loss / n_batch, correct, size, 100. * correct / size
        ))

        return float(correct) / float(size)
