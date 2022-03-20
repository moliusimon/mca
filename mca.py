import torch
import torch.nn as nn
import numpy as np


def atanh(x):
    return 0.5*torch.log((1+x)/(1-x))


class MCALoss(nn.Module):
    def __init__(self, n_smp, projections=1, mca_weight=50):
        super(MCALoss, self).__init__()
        self.projections = projections
        self.mca_weight = mca_weight

        self.mask_s = torch.cat((torch.ones([n_smp]), torch.zeros([n_smp])), 0).cuda()
        self.mask_t = 1 - self.mask_s

    def forward(self, xs, xt, iteration):
        """ Direct CDE ratio optimization """

        # If data has spatial resolution, turn each location into a sample
        if len(xs.shape) == 4:
            xs, xt = self._transform_img_tensors(xs, xt)

        n_smp = xs.shape[0]
        self.mask_s = torch.cat((torch.ones([n_smp]), torch.zeros([n_smp])), 0).cuda()
        self.mask_t = 1 - self.mask_s

        # Compute sample re-weighting & concatenate samples
        smp_w = self._get_sample_weights(xs, xt)
        x = torch.cat((xs, xt), 0)

        # Generate random hyper-plane normals
        normals = torch.normal(0, 1, [self.projections, x.shape[0], x.shape[1]]).cuda()
        normals = normals / torch.norm(normals, p=2, dim=-1, keepdim=True)

        # Compute 1-d space projections and extract signs wrt hyper-planes
        dots = torch.einsum('ijk,lk->ijl', normals, x)
        dists = dots - torch.diagonal(dots, dim1=-2, dim2=-1).detach()[..., None]
        signs = torch.einsum('ijk,k->ijk', torch.tanh(dists), smp_w)

        # Compute Cumulative Density Estimations on 1-d space projections (clamp to prevent zero probabilities)
        s_cde = torch.clamp(torch.sum(signs * self.mask_s, -1) / xs.shape[0], 1e-4 - 1)
        t_cde = torch.clamp(torch.sum(signs * self.mask_t, -1) / xt.shape[0], 1e-4 - 1)

        # Normalize CDE values
        s_cde, t_cde = 1 + s_cde, 1 + t_cde
        sum_c = s_cde + t_cde
        s_cde, t_cde = s_cde / sum_c, t_cde / sum_c

        # Compute & return entropy maximization loss
        mca_w = self._calc_weight(iteration)
        mca_loss = torch.mean(s_cde * torch.log(s_cde) + t_cde * torch.log(t_cde))

        return mca_w * mca_loss

    def _calc_weight(self, iteration):
        p = iteration / 10000.
        # return self.mca_weight * (2. / (1 + np.exp(-p)) - 1)
        return self.mca_weight * (2. / (1 + np.exp(-10 * p)) - 1)

    def _get_sample_weights(self, xs, xt):
        xt_c = xt - torch.mean(xt, 0, keepdim=True)
        w = torch.ones(xs.shape[0]).cuda()

        for i in range(10):
            # Center source data (TODO: Fix weight normalization?)
            xs_c = xs - torch.matmul(w / torch.norm(w, 1), xs)
            # xs_c = xs - torch.matmul(w / xs.shape[0], xs)

            # Compute sample covariance matrices
            cov_ss = torch.matmul(xs_c, xs_c.T)
            cov_st = torch.matmul(xs_c, xt_c.T)

            # # Compute A and B, source weights
            a = (cov_ss ** 2 * (1 + torch.normal(0, 0.3, cov_ss.shape).cuda())) / xs.shape[0]
            b = torch.mean(cov_st ** 2 * (1 + torch.normal(0, 0.3, cov_st.shape).cuda()), -1)

            # Compute A and B, source weights
            # a = cov_ss ** 2 / xs.shape[0]
            # b = torch.mean(cov_st ** 2, axis=-1)

            # Inject noise to A and regress
            a_inv = torch.pinverse(a * torch.normal(1, 0.3, cov_ss.shape, device='cuda'))
            # w = torch.matmul(a_inv, b)
            w = torch.matmul(a_inv + a_inv.T, b) / 2

        w = xs.shape[0] * w / torch.norm(w, 1)
        return torch.cat((w, torch.ones(xt.shape[0]).cuda()), 0).detach()

    # def _get_sample_weights(self, xs, xt):
    #     xt_c = xt - torch.mean(xt, 0, keepdim=True)
    #     w = torch.ones(xs.shape[0]).cuda() / xs.shape[0]
    #
    #     for i in range(5):
    #         # Center source data
    #         xs_c = xs - torch.matmul(w, xs)
    #
    #         # Compute sample covariance matrices
    #         cov_ss = torch.matmul(xs_c, xs_c.T)
    #         cov_st = torch.matmul(xs_c, xt_c.T)
    #
    #         # # Compute A and B, source weights
    #         # a = (cov_ss ** 2 * (1 + torch.normal(0, 0.3, cov_ss.shape).cuda())) / xs.shape[0]
    #         # b = torch.mean(cov_st ** 2 * (1 + torch.normal(0, 0.3, cov_st.shape).cuda()), -1)
    #
    #         # Compute A and B, source weights
    #         a = cov_ss ** 2 / xs.shape[0]
    #         b = torch.mean(cov_st ** 2, axis=-1)
    #
    #         # Inject noise to A and regress
    #         a_inv = torch.pinverse(a) * torch.normal(1, 0.3, cov_ss.shape, device='cuda')
    #         # w = torch.matmul(a_inv, b)
    #         w = torch.matmul(a_inv + a_inv.T, b)
    #         w = w / torch.norm(w, 1)
    #
    #     return torch.cat((
    #         xs.shape[0] * w,
    #         torch.ones(xt.shape[0]).cuda()
    #     ), 0).detach()

    @staticmethod
    def _truncated_lsq(a, b, var=0.90):
        u, s, v = torch.svd(a)
        s_inv = (torch.cumsum(s, -1) / torch.sum(s) <= var) / s
        return torch.matmul(v, torch.matmul(u.T * s_inv[:, None], b))

    @staticmethod
    def _transform_img_tensors(s, t, max_smp=128):
        """
        4-d tensors are considered as image tensors and processed such that each spatial
        location is an instance. Furthermore, a max. number of samples is randomly drawn
        from each image.
        """

        # Transform pixel locations into samples
        s = s.permute(0, 2, 3, 1).reshape(-1, s.shape[1])
        t = t.permute(0, 2, 3, 1).reshape(-1, t.shape[1])

        # Randomly sample source & target (if required)
        s = s[torch.randperm(s.shape[0])[:max_smp]] if s.shape[0] > max_smp else s
        t = t[torch.randperm(t.shape[0])[:max_smp]] if t.shape[0] > max_smp else t

        return s, t


def pseudo_labelled_cross_entropy_loss(xs_logits, xt_logits, xs_labels):
    # Compute cross-entropy loss on source data
    losses = -nn.functional.log_softmax(xs_logits, -1)[range(xs_logits.shape[0]), xs_labels]
    loss = torch.mean(losses)

    # If loss on source data is below specific value
    if loss.item() < 0.2:
        # Compute loss on target dataset by assigning labels to high certainty samples
        xs_losses = -torch.max(nn.functional.log_softmax(xt_logits, -1), dim=-1).values
        losses = torch.cat((losses, xs_losses[xs_losses < 0.005]), 0)
        loss = torch.mean(losses)

    return loss
