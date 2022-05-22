import argparse
import os
import math
import time
import random
import nibabel
import numpy as np
import yaml
import torch
from torch.utils.data import DataLoader
from time import strftime
from datasets import tokenize, DiffusionDataset
from models import SimpleNet
from utils import find_file, get_mask, get_normalised_image
from utils import color_codes, time_to_string


"""
> Arguments
"""


def parse_inputs():
    parser = argparse.ArgumentParser(
        description='Train models with incremental learning approaches and '
                    'test them to obtain timeseries metrics of simple'
                    'overlap concepts.'
    )

    # Mode selector
    parser.add_argument(
        '-i', '--input-config',
        dest='config', default='/data/IncrementalLearning/activity_dual.yml',
        help='Path to the file with the configuration for the experiment.'
    )
    options = vars(parser.parse_args())

    return options


"""
> Data functions
"""


def get_subject(config, p_path):
    roi = get_mask(find_file(config['roi'], p_path))
    image_name = find_file(config['image'], p_path)
    raw_image = np.moveaxis(nibabel.load(image_name).get_fdata(), -1, 0)
    norm_image = get_normalised_image(raw_image, roi)
    bvecs_file = find_file(config['bvecs'], p_path)
    bvals_file = find_file(config['bvals'], p_path)
    with open(bvecs_file) as f:
        bvecs = []
        for s_i in f.readlines():
            bvecs.append([float(bvec) for bvec in s_i.split(' ')])
        bvecs_array = np.stack(bvecs, axis=-1)
    with open(bvals_file) as f:
        for s_i in f.readlines():
            bvals_array = np.array([int(bval) for bval in s_i.split('  ')])

    return norm_image, roi, bvecs_array, bvals_array


def get_data(config, subjects):
    # Init
    d_path = config['path']
    images = []
    rois = []
    direction_list = []
    bvalue_list = []

    for pi, p in enumerate(subjects):
        p_path = os.path.join(d_path, p)
        image, roi, directions, bvalues = get_subject(config, p_path)
        rois.append(roi)
        images.append(image)
        direction_list.append(directions)
        bvalue_list.append(bvalues)

    return images, rois, direction_list, bvalue_list


"""
> Network functions
"""


def train(config, net, training, validation, model_name, verbose=0):
    """

    :param config:
    :param net:
    :param training:
    :param validation:
    :param model_name:
    :param verbose:
    """
    # Init
    path = config['model_path']
    epochs = config['epochs']
    patience = config['patience']

    try:
        net.load_model(os.path.join(path, model_name))
    except IOError:

        if verbose > 1:
            print('Preparing the training datasets / dataloaders')

        # Training
        if verbose > 1:
            print('< Training dataset >')
        itrain, rtrain, dtrain, btrain = get_data(config, training)
        if 'test_patch' in config and 'test_overlap' in config:
            train_dataset = DiffusionDataset(
                itrain, rtrain, dtrain, btrain,
                patch_size=config['test_patch'],
                overlap=config['train_overlap']
            )
        elif 'test_patch' in config:
            train_dataset = DiffusionDataset(
                itrain, rtrain, dtrain, btrain,
                patch_size=config['test_patch']
            )
        else:
            train_dataset = DiffusionDataset(
                itrain, rtrain, dtrain, btrain
            )

        if verbose > 1:
            print('Dataloader creation <with validation>')
        train_loader = DataLoader(
            train_dataset, config['train_batch'], True, num_workers=8
        )

        # Validation (training cases)
        if verbose > 1:
            print('< Validation dataset >')
        if training == validation:
            ival, rval, dval, bval = itrain, rtrain, dtrain, btrain
        else:
            ival, rval, dval, bval = get_data(config, validation)
        if 'test_patch' in config and 'test_overlap' in config:
            val_dataset = DiffusionDataset(
                ival, rval, dval, bval,
                patch_size=config['test_patch'],
                overlap=config['train_overlap']
            )
        elif 'test_patch' in config:
            val_dataset = DiffusionDataset(
                ival, rval, dval, bval,
                patch_size=config['test_patch']
            )
        else:
            val_dataset = DiffusionDataset(
                ival, rval, dval, bval
            )

        if verbose > 1:
            print('Dataloader creation <val>')
        val_loader = DataLoader(
            val_dataset, config['test_batch'], num_workers=8
        )

        if verbose > 1:
            print(
                'Training / validation samples samples = '
                '{:02d} / {:02d}'.format(
                    len(train_dataset), len(val_dataset)
                )
            )

        net.fit(
            train_loader, val_loader, epochs=epochs, patience=patience
        )
        net.save_model(os.path.join(path, model_name))


def test(
    config, seed, net, base_name, testing_subjects, verbose=0
):
    # Init
    mask_name = '{:}.s{:05d}.nii.gz'.format(base_name, seed)
    test_start = time.time()
    for sub_i, subject in enumerate(testing_subjects):
        tests = len(testing_subjects) - sub_i
        test_elapsed = time.time() - test_start
        test_eta = tests * test_elapsed / (sub_i + 1)
        if verbose > 0:
            print(
                '\033[KTesting subject {:} ({:d}/{:d}) {:} ETA {:}'.format(
                    subject, sub_i + 1, len(testing_subjects),
                    time_to_string(test_elapsed),
                    time_to_string(test_eta),
                ), end='\r'
            )

        d_path = os.path.join(config['path'], subject)
        p_path = os.path.join(d_path, subject)
        mask_path = os.path.join(p_path, mask_name)
        hr_image, _, directions, bvalues = get_subject(config, p_path)
        lr_image = hr_image[:22, ...]
        token = tokenize(hr_image, directions, bvalues)
        key_data = token[:, :22, ...]
        query_data = token[:-1, 22:, ...]
        extra_image = net.patch_inference(
            (key_data, query_data), config['test_patch'], config['test_batch'],
            sub_i, len(testing_subjects), test_start
        )
        hr_prediction = np.concatenate([lr_image, extra_image])
        image_nii = nibabel.load(find_file(config['image'], p_path))
        prediction_nii = nibabel.Nifti1Image(
            hr_prediction, image_nii.get_qform(), image_nii.header
        )
        prediction_nii.to_filename(mask_path)


"""
> Dummy main function
"""


def main():
    # Init
    c = color_codes()
    options = parse_inputs()
    with open(options['config'], 'r') as stream:
        try:
            config = yaml.load(stream, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            print(exc)
    n_folds = config['folds']
    val_split = config['val_split']
    model_path = config['model_path']
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    d_path = config['path']
    seeds = config['seeds']

    print(
        '{:}[{:}] {:}<Incremental learning framework>{:}'.format(
            c['c'], strftime("%H:%M:%S"), c['y'], c['nc']
        )
    )

    # We want a common starting point
    subjects = sorted([
        subject for subject in os.listdir(d_path)
        if os.path.isdir(os.path.join(d_path, subject))
    ])

    # Main loop with all the seeds
    for test_n, seed in enumerate(seeds):
        print(
            '{:}[{:}] {:}Starting cross-validation (model: simple){:}'
            ' (seed {:d}){:}'.format(
                c['clr'] + c['c'], strftime("%H:%M:%S"), c['g'],
                c['nc'] + c['y'], seed, c['nc']
            )
        )
        # Network init (random weights)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        net = SimpleNet()
        starting_model = os.path.join(
            model_path, 'simple-quad22-start.s{:05d}.pt'.format(seed)
        )
        net.save_model(starting_model)
        n_param = sum(
            p.numel() for p in net.parameters() if p.requires_grad
        )

        print(
            '{:}Testing initial weights{:} - {:02d}/{:02d} '
            '({:} parameters)'.format(
                c['clr'] + c['c'], c['nc'], test_n + 1, len(config['seeds']),
                c['b'] + str(n_param) + c['nc']
            )
        )

        random.shuffle(subjects)
        n_test = int(math.ceil(len(subjects) / n_folds))
        # Cross-validation loop
        for i in range(n_folds):
            testing = subjects[i * n_test:(i + 1) * n_test]
            not_testing = subjects[(i + 1) * n_test:] + subjects[0:i * n_test]
            if val_split > 0:
                validation = not_testing[:len(not_testing) * val_split]
                training = not_testing[len(not_testing) * val_split:]
            else:
                training = validation = not_testing

            fold_name = os.path.join(
                model_path,
                'simple-quad22-start.n{:05d}.s{:05d}.pt'.format(i, seed)
            )
            train(config, net, training, validation, fold_name, 2)
            test(config, seed, net, 'simple-quad22', testing, 2)


if __name__ == '__main__':
    main()
