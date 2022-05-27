import time
import itertools
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from base import BaseModel, ResConv3dBlock
from base import MultiheadedAttention, MultiheadedPairedAttention
from utils import time_to_string, to_torch_var


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
            MultiheadedAttention(4, f_att, heads, 3)
            for f_att in self.encoder_filters
        ])
        self.key_encoder.to(self.device)
        self.query_encoder = nn.ModuleList([
            MultiheadedAttention(3, f_att, heads, 3)
            for f_att in self.encoder_filters
        ])
        self.query_encoder.to(self.device)
        self.bottleneck = MultiheadedPairedAttention(
            4, 3, self.encoder_filters[-1], heads, 3
        )
        self.decoder = nn.ModuleList([
            MultiheadedAttention(3, f_att, heads, 3)
            for f_att in self.decoder_filters
        ])
        self.decoder.to(self.device)
        self.final = nn.Conv3d(3, 1, 1)

        # <Loss function setup>
        self.train_functions = [
            {
                'name': 'mse',
                'weight': 1,
                'f': F.mse_loss
            }
        ]

        self.val_functions = [
            {
                'name': 'mse',
                'weight': 1,
                'f': F.mse_loss
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


class CroppedNet(SimpleNet):
    def __init__(
            self,
            encoder_filters=None, decoder_filters=None, heads=32,
            device=torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            ),
            verbose=0,
    ):
        super().__init__(
            encoder_filters, decoder_filters, heads,
            device, 0
        )
        self.decoder = nn.ModuleList([
            MultiheadedAttention(3, f_att, heads, 1)
            for f_att in self.decoder_filters
        ])

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

    def patch_inference(
        self, data, patch_size, batch_size, case=0, n_cases=1, t_start=None
    ):
        # Init
        self.eval()

        # Init
        t_in = time.time()
        if t_start is None:
            t_start = t_in

        # This branch is only used when images are too big. In this case
        # they are split in patches and each patch is trained separately.
        # Currently, the image is partitioned in blocks with no overlap,
        # however, it might be a good idea to sample all possible patches,
        # test them, and average the results. I know both approaches
        # produce unwanted artifacts, so I don't know.
        # Initial results. Filled to 0.
        if isinstance(data, tuple):
            data_shape = data[1].shape[:1] + data[1].shape[-3:]
        else:
            data_shape = data.shape[:1] + data.shape[-3:]
        seg = np.zeros(data_shape)
        counts = np.zeros(data_shape)

        # The following lines are just a complicated way of finding all
        # the possible combinations of patch indices.
        steps = [
            list(
                range(0, lim - patch_size, patch_size - self.crop * 2)
            ) + [lim - patch_size]
            for lim in data_shape[1:]
        ]

        steps_product = list(itertools.product(*steps))
        batches = range(0, len(steps_product), batch_size)
        n_batches = len(batches)

        # The following code is just a normal test loop with all the
        # previously computed patches.
        for bi, batch in enumerate(batches):
            # Here we just take the current patch defined by its slice
            # in the x and y axes. Then we convert it into a torch
            # tensor for testing.
            in_slices = [
                (
                    slice(xi, xi + patch_size),
                    slice(xj, xj + patch_size),
                    slice(xk, xk + patch_size)
                )
                for xi, xj, xk in steps_product[batch:(batch + batch_size)]
            ]
            out_slices = [
                (
                    slice(xi + self.crop, xi + patch_size - self.crop),
                    slice(xj + self.crop, xj + patch_size - self.crop),
                    slice(xk + self.crop, xk + patch_size - self.crop)
                )
                for xi, xj, xk in steps_product[batch:(batch + batch_size)]
            ]

            # Testing itself.
            with torch.no_grad():
                if isinstance(data, list) or isinstance(data, tuple):
                    batch_cuda = tuple(
                        torch.stack([
                            torch.from_numpy(
                                x_i[
                                    slice(None), slice(None),
                                    xslice, yslice, zslice
                                ]
                            ).type(torch.float32).to(self.device)
                            for xslice, yslice, zslice in in_slices
                        ])
                        for x_i in data
                    )
                    seg_out = self(*batch_cuda)
                else:
                    batch_cuda = torch.stack([
                        torch.from_numpy(
                            data[
                                slice(None), slice(None),
                                xslice, yslice, zslice
                            ]
                        ).type(torch.float32).to(self.device)
                        for xslice, yslice, zslice in in_slices
                    ])
                    seg_out = self(batch_cuda)
                torch.cuda.empty_cache()

            # Then we just fill the results image.
            for si, (xslice, yslice, zslice) in enumerate(out_slices):
                counts[slice(None), xslice, yslice, zslice] += 1
                seg_bi = seg_out[si, :].cpu().numpy()
                seg[slice(None), xslice, yslice, zslice] += seg_bi

            # Printing
            self.print_batch(bi, n_batches, case, n_cases, t_start, t_in)

        seg /= counts

        return seg
