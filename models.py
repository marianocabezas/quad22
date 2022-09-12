import time
from functools import partial
import torch
from torch import nn
import torch.nn.functional as F
from base import BaseModel
from base import MultiheadedAttention, SelfAttentionBlock, ResConv3dBlock
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
            MultiheadedAttention(3, 3, f_att, heads, 3)
            for f_att in self.encoder_filters
        ])
        self.query_encoder.to(self.device)
        self.bottleneck = MultiheadedAttention(
            4, 3, self.encoder_filters[-1], heads, 1
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
            encoder_filters=None, heads=32,
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
            self.encoder_filters = [16, 32, 64, 128, 256, 512]
        else:
            self.encoder_filters = encoder_filters
        self.decoder_filters = self.encoder_filters[-2::-1]

        self.epoch = 0
        self.t_train = 0
        self.t_val = 0
        self.device = device

        # <Parameter setup>
        enc_in = [features] + self.encoder_filters[:-1]
        self.encoder = nn.ModuleList([
            SelfAttentionBlock(f_in, f_out, heads, 3)
            for f_in, f_out in zip(enc_in, self.encoder_filters)
        ])
        self.encoder.to(self.device)

        dec_skip = self.encoder_filters[-2::-1]
        dec_up = self.decoder_filters[:-1]
        self.decoder = nn.ModuleList([
            SelfAttentionBlock(f_out + f_up, f_out, heads, 3)
            for f_out, f_up in zip(dec_skip, dec_up)
        ])
        self.decoder.to(self.device)

        final_feat = self.encoder_filters[0]
        self.pred_token = nn.Parameter(
            torch.rand((1, 1, final_feat, 1, 1, 1)), requires_grad=True
        )
        self.final_tf = SelfAttentionBlock(final_feat, final_feat, heads, 3)
        self.final = nn.Conv3d(final_feat, 6, 1)
        self.final.to(self.device)

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
        # crop_slice = slice(self.crop, -self.crop)
        # loss = F.mse_loss(
        #     prediction, target[..., crop_slice, crop_slice, crop_slice]
        # )
        return F.mse_loss(prediction, target)

    def reset_optimiser(self):
        super().reset_optimiser()
        model_params = filter(lambda p: p.requires_grad, self.parameters())
        self.optimizer_alg = torch.optim.Adam(model_params)

    def forward(self, data, bvecs):
        positional = torch.matmul(
            bvecs, bvecs.transpose(1, 2)
        ).squeeze(0)
        skip_inputs = []
        for sa in self.encoder[:-1]:
            sa.to(self.device)
            print('Input', data.shape, positional.shape)
            data = sa(data, positional)
            skip_inputs.append(data)
            data_flat = F.max_pool3d(data.flatten(0, 1), 2)
            print('Post-SA', data.shape, data_flat.shape)
            data = data_flat.view((data.shape[0], -1) + data_flat.shape[1:])
        self.encoder[-1].to(self.device)
        data = self.encoder[-1](data, positional)
        for sa, i in zip(self.decoder, skip_inputs[::-1]):
            sa.to(self.device)
            data_flat = F.interpolate(data.flatten(0, 1), size=i.size()[2:])
            data = sa(
                data_flat.view(
                    (data.shape[0], -1) + self.encoder[-1].shape[1:]
                ), positional
            )

        pred_token = self.pred_token.expand(
            (data.shape[0], 1, -1) + data.shape[3:]
        )
        data = torch.cat([data, pred_token], dim=1)

        self.final_tf.to(self.device)
        data = self.final_tf(data)
        self.final.to(self.device)
        return self.final(data[:, -1, ...])


class TensorUnet(BaseModel):
    def __init__(
            self,
            encoder_filters=None, decoder_filters=None,
            device=torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            ),
            features=21,
            verbose=0,
    ):
        super().__init__()
        self.init = False
        # Init values
        if encoder_filters is None:
            encoder_filters = [4, 8, 16, 32, 64, 128]
        if decoder_filters is None:
            decoder_filters = encoder_filters[:0:-1]
        if decoder_filters >= encoder_filters:
            shift = len(decoder_filters) - len(encoder_filters) + 1
            extra_filters = decoder_filters[:shift]
            encoder_filters += extra_filters[::-1]
        if len(decoder_filters) < (len(encoder_filters) - 1):
            shift = len(encoder_filters) - len(decoder_filters)
            extra_filters = encoder_filters[:shift:-1]
            decoder_filters = extra_filters + decoder_filters
        self.epoch = 0
        self.t_train = 0
        self.t_val = 0
        self.device = device

        # <Parameter setup>
        # Init
        block_partial = partial(
            ResConv3dBlock, kernel=3, norm=norm_f, activation=nn.ReLU
        )

        # Down path
        # We'll use the partial and fill it with the channels for input and
        # output for each level.
        self.encoder = nn.ModuleList([
            block_partial(f_in, f_out) for f_in, f_out in zip(
                [features] + encoder_filters[:-2], encoder_filters[:-1]
            )
        ])
        self.encoder.to(self.device)

        # Bottleneck
        self.bottleneck = block_partial(
            encoder_filters[-2], encoder_filters[-1]
        )

        # Up path
        # Now we'll do the same we did on the down path, but mirrored. We also
        # need to account for the skip connections, that's why we sum the
        # channels for both outputs. That basically means that we are
        # concatenating with the skip connection, and not adding to it.
        skip_filters = encoder_filters[-2::-1]
        up_filters = [encoder_filters[-1]] + decoder_filters[:-1]
        deconv_in = [
            skip_i + up_i for skip_i, up_i in zip(skip_filters, up_filters)
        ]
        self.decoder = nn.ModuleList([
            block_partial(f_in, f_out) for f_in, f_out in zip(
                deconv_in, decoder_filters
            )
        ])
        self.decoder.to(self.device)
        self.final = nn.Conv3d(decoder_filters[-1], 6, 1)
        self.final.to(self.device)

        # <Loss function setup>
        self.train_functions = [
            {
                'name': 'mse',
                'weight': 1e3,
                'f': self.mse_loss
            }
        ]

        self.val_functions = [
            {
                'name': 'mse',
                'weight': 1e3,
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

    def reset_optimiser(self):
        super().reset_optimiser()
        model_params = filter(lambda p: p.requires_grad, self.parameters())
        self.optimizer_alg = torch.optim.Adam(model_params)

    def mse_loss(self, prediction, target_tensors):
        roi, target = target_tensors
        loss = F.mse_loss(prediction * roi, target * roi)
        return loss

    def forward(self, data, bvecs):
        # This is a dirty hack to reuse the same dataset we used for
        # self-attention.
        data = torch.squeeze(data, 2)
        # positional = torch.matmul(bvecs, bvecs.transpose(1, 2))
        # new_shape = positional.shape[:1] + (1,) * 3 + positional.shape[-2:]
        # positional = positional.view(new_shape)
        # We need to keep track of the convolutional outputs, for the skip
        # connections.
        down_inputs = []
        for c in self.encoder:
            c.to(self.device)
            data = c(data)
            down_inputs.append(data)
            # Remember that pooling is optional
            data = F.max_pool3d(data, 2)

        self.bottleneck.to(self.device)
        data = self.bottleneck(data)

        for d, i in zip(self.decoder, down_inputs[::-1]):
            d.to(self.device)
            # Remember that pooling is optional
            data = d(
                torch.cat(
                    (F.interpolate(data, size=i.size()[2:]), i),
                    dim=1
                )
            )

        self.final.to(self.device)
        return self.final(data)
