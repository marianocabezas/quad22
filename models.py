import time
import torch
from torch import nn
import torch.nn.functional as F
from base import BaseModel
from base import MultiheadedAttention
from utils import time_to_string


def norm_f(n_f):
    return nn.GroupNorm(n_f // 4, n_f)


def print_batch(pi, n_patches, i, n_cases, t_in, t_case_in):
    init_c = '\033[38;5;238m'
    percent = 25 * (pi + 1) // n_patches
    progress_s = ''.join(['█'] * percent)
    remainder_s = ''.join([' '] * (25 - percent))

    t_out = time.time() - t_in
    t_case_out = time.time() - t_case_in
    time_s = time_to_string(t_out)

    t_eta = (t_case_out / (pi + 1)) * (n_patches - (pi + 1))
    eta_s = time_to_string(t_eta)
    pre_s = '{:}Case {:03d}/{:03d} ({:03d}/{:03d} - {:06.2f}%) [{:}{:}]' \
            ' {:} ETA: {:}'
    batch_s = pre_s.format(
        init_c, i + 1, n_cases, pi + 1, n_patches, 100 * (pi + 1) / n_patches,
        progress_s, remainder_s, time_s, eta_s + '\033[0m'
    )
    print('\033[K', end='', flush=True)
    print(batch_s, end='\r', flush=True)


class SimpleNet(BaseModel):
    def __init__(
            self,
            encoder_filters=None, decoder_filters=None, heads=32,
            device=torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            ),
            verbose=0,
    ):
        super().__init__()
        self.init = False
        # Init values
        if encoder_filters is None:
            self.encoder_filters = [4, 4, 4, 4, 4, 4]
        else:
            self.encoder_filters = encoder_filters
        if decoder_filters is None:
            self.decoder_filters = self.encoder_filters[::-1]
        else:
            self.decoder_filters = decoder_filters
        self.epoch = 0
        self.t_train = 0
        self.t_val = 0
        self.device = device

        # <Parameter setup>
        self.key_encoder = nn.ModuleList([
            MultiheadedAttention(4, 4, f_att, heads, 3)
            for f_att in self.encoder_filters
        ])
        self.key_encoder.to(self.device)
        self.query_encoder = nn.ModuleList([
            MultiheadedAttention(3, f_att, heads, 3)
            for f_att in self.encoder_filters
        ])
        self.query_encoder.to(self.device)
        self.bottleneck = MultiheadedAttention(
            4, 3, self.encoder_filters[-1], heads, 3
        )
        self.bottleneck.to(self.device)
        self.decoder = nn.ModuleList([
            MultiheadedAttention(3, 3, f_att, heads, 1)
            for f_att in self.decoder_filters
        ])
        self.decoder.to(self.device)
        self.final = nn.Conv3d(3, 1, 1)
        self.final.to(self.device)

        self.crop = len(self.encoder_filters)
        # <Loss function setup>
        self.train_functions = [
            {
                'name': 'mse',
                'weight': 1,
                'f': self.mse_loss
            }
        ]

        self.val_functions = [
            {
                'name': 'mse',
                'weight': 1,
                'f': self.mse_loss
            }
        ]

        # <Optimizer setup>
        # We do this last step after all parameters are defined
        model_params = filter(lambda p: p.requires_grad, self.parameters())
        self.optimizer_alg = torch.optim.Adam(model_params)
        if verbose > 1:
            print(
                'Network created on device {:} with training losses '
                '[{:}] and validation losses [{:}]'.format(
                    self.device,
                    ', '.join([tf['name'] for tf in self.train_functions]),
                    ', '.join([vf['name'] for vf in self.val_functions])
                )
            )

    def mse_loss(self, prediction, target):
        crop_slice = slice(self.crop, -self.crop)
        loss = F.mse_loss(
            prediction, target[..., crop_slice, crop_slice, crop_slice]
        )
        return loss

    def reset_optimiser(self):
        super().reset_optimiser()
        model_params = filter(lambda p: p.requires_grad, self.parameters())
        self.optimizer_alg = torch.optim.Adam(model_params)

    def forward(self, key, query):
        for k_sa, q_sa in zip(self.key_encoder, self.query_encoder):
            k_sa.to(self.device)
            key = k_sa(key)
            q_sa.to(self.device)
            query = q_sa(query)
        self.bottleneck.to(self.device)
        feat = self.bottleneck(key, query)
        for sa in self.decoder:
            sa.to(self.device)
            feat = sa(feat)

        feat_flat = feat.flatten(0, 1)
        self.final.to(self.device)
        dwi_flat = self.final(feat_flat)

        return dwi_flat.view(feat.shape[:2] + feat.shape[3:])


class PositionalNet(BaseModel):
    def __init__(
            self,
            encoder_filters=None, decoder_filters=None, heads=32,
            device=torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            ),
            features=1,
            verbose=0,
    ):
        super().__init__()
        self.init = False
        # Init values
        if encoder_filters is None:
            self.encoder_filters = [4, 4, 4, 4, 4, 4]
        else:
            self.encoder_filters = encoder_filters
        if decoder_filters is None:
            self.decoder_filters = self.encoder_filters[::-1]
        else:
            self.decoder_filters = decoder_filters
        self.epoch = 0
        self.t_train = 0
        self.t_val = 0
        self.device = device

        # <Parameter setup>
        self.encoder = nn.ModuleList([
            MultiheadedAttention(features, features, f_att, heads, 3)
            for f_att in self.encoder_filters
        ])
        self.encoder.to(self.device)
        self.decoder = nn.ModuleList([
            MultiheadedAttention(features, features, f_att, heads, 1)
            for f_att in self.decoder_filters
        ])
        self.decoder.to(self.device)
        self.final = nn.Conv3d(features, 6, 1)
        self.final.to(self.device)

        self.crop = len(self.encoder_filters)
        # <Loss function setup>
        self.train_functions = [
            {
                'name': 'mse',
                'weight': 1,
                'f': self.mse_loss
            }
        ]

        self.val_functions = [
            {
                'name': 'mse',
                'weight': 1,
                'f': self.mse_loss
            }
        ]

        # <Optimizer setup>
        # We do this last step after all parameters are defined
        model_params = filter(lambda p: p.requires_grad, self.parameters())
        self.optimizer_alg = torch.optim.Adam(model_params)
        if verbose > 1:
            print(
                'Network created on device {:} with training losses '
                '[{:}] and validation losses [{:}]'.format(
                    self.device,
                    ', '.join([tf['name'] for tf in self.train_functions]),
                    ', '.join([vf['name'] for vf in self.val_functions])
                )
            )

    def mse_loss(self, prediction, target):
        crop_slice = slice(self.crop, -self.crop)
        loss = F.mse_loss(
            prediction, target[..., crop_slice, crop_slice, crop_slice]
        )
        return loss

    def reset_optimiser(self):
        super().reset_optimiser()
        model_params = filter(lambda p: p.requires_grad, self.parameters())
        self.optimizer_alg = torch.optim.Adam(model_params)

    def forward(self, data, bvecs):
        positional = torch.matmul(bvecs[0], bvecs[0].transpose())
        for sa in zip(self.encoder):
            sa.to(self.device)
            data = sa(data, None, positional)
        for sa in self.decoder:
            sa.to(self.device)
            data = sa(data)

        feat_flat = data.flatten(0, 1)
        self.final.to(self.device)
        dwi_flat = self.final(feat_flat)

        return torch.sum(dwi_flat.view(data.shape[:2] + data.shape[3:]), dim=1)
