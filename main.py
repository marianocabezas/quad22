import argparse
import importlib
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
from datasets import tokenize
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
    image_name = find_file(config['image'], p_path)
    raw_image = np.moveaxis(nibabel.load(image_name).get_fdata(), -1, 0)
    tensor = nibabel.load(find_file(config['tensor'], p_path)).get_fdata()
    tensor = np.moveaxis(tensor, -1, 0)
    roi = get_mask(find_file(config['roi'], p_path))
    bvecs_file = find_file(config['bvecs'], p_path)
    bvals_file = find_file(config['bvals'], p_path)
    with open(bvecs_file) as f:
        bvecs = []
        for s_i in f.readlines():
            bvecs.append([float(bvec) for bvec in s_i.split(' ')])
        bvecs_array = np.stack(bvecs, axis=-1)[1:, :]
    with open(bvals_file) as f:
        for s_i in f.readlines():
            bvals_array = np.array([int(bval) for bval in s_i.split('  ')])[1:]
    bvals_array = np.expand_dims(bvals_array, axis=(1, 2, 3))
    b0 = raw_image[:1]
    dmri = raw_image[1:]
    invalid_b0 = b0 <= 0
    invalid_dmri = dmri <= 0
    b0[invalid_b0] = 1e-6
    log_b0 = np.log(b0)
    log_b0[invalid_b0] = 0
    dmri[invalid_dmri] = 1e-6
    log_dmri = np.log(dmri)
    log_dmri[invalid_dmri] = 0
    norm_image = (log_b0 - log_dmri) / bvals_array

    return norm_image, log_b0, tensor, roi, bvecs_array, bvals_array


def get_data(config, subjects):
    # Init
    d_path = config['path']
    images = []
    tensors = []
    rois = []
    direction_list = []
    load_start = time.time()
    for pi, p in enumerate(subjects):
        tests = len(subjects) - pi
        test_elapsed = time.time() - load_start
        test_eta = tests * test_elapsed / (pi + 1)
        print(
            '\033[KLoading subject {:} ({:d}/{:d}) {:} ETA {:}'.format(
                p, pi + 1, len(subjects),
                time_to_string(test_elapsed),
                time_to_string(test_eta),
            ), end='\r'
        )
        p_path = os.path.join(d_path, p)
        image, _, tensor, roi, directions, bvalues = get_subject(
            config, p_path
        )

        images.append(image)
        tensors.append(tensor)
        rois.append(roi)
        direction_list.append(directions)

    return images, tensors, rois, direction_list


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
            print('\033[KPreparing the training datasets / dataloaders')
        datasets = importlib.import_module('datasets')
        dataset_class = getattr(datasets, config['dataset'])
        # Training
        if verbose > 1:
            print('< Training dataset >')
        itrain, ttrain, rtrain, dtrain = get_data(config, training)
        if 'test_patch' in config and 'test_overlap' in config:
            train_dataset = dataset_class(
                itrain, ttrain, rtrain, dtrain,
                patch_size=config['test_patch'],
                overlap=config['train_overlap']
            )
        elif 'test_patch' in config:
            train_dataset = dataset_class(
                itrain, ttrain, rtrain, dtrain,
                patch_size=config['test_patch']
            )
        else:
            train_dataset = dataset_class(
                itrain, ttrain, rtrain, dtrain,
            )

        if verbose > 1:
            print('\033[KDataloader creation <with validation>')
        train_loader = DataLoader(
            train_dataset, config['train_batch'], True, num_workers=8
        )

        # Validation (training cases)
        if verbose > 1:
            print('< Validation dataset >')
        if training == validation:
            ival, tval, rval, dval = itrain, ttrain, rtrain, dtrain
        else:
            ival, tval, rval, dval = get_data(config, validation)
        if 'test_patch' in config and 'test_overlap' in config:
            val_dataset = dataset_class(
                ival, tval, rval, dval, shift=False,
                patch_size=config['test_patch'],
                overlap=config['train_overlap']
            )
        elif 'test_patch' in config:
            val_dataset = dataset_class(
                ival, tval, rval, dval, shift=False,
                patch_size=config['test_patch']
            )
        else:
            val_dataset = dataset_class(
                ival, tval, rval, dval, shift=False,
            )

        if verbose > 1:
            print('\033[KDataloader creation <val>')
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

        p_path = os.path.join(config['path'], subject)
        mask_path = os.path.join(p_path, mask_name)
        hr_image, log_b0, tensor, roi, directions, bvalues = get_subject(
            config, p_path
        )
        lr_image = hr_image[:21, ...]
        roi = np.expand_dims(roi, 0).astype(np.float32)
        if config['tokenize']:
            token = tokenize(hr_image, directions)
            input_data = (token[:21, ...], token[21:, :-1, ...])
            extra_image = net.patch_inference(
                input_data, config['test_patch'], config['test_batch'],
                None, sub_i, len(testing_subjects),
                test_start
            )
            image = np.concatenate([lr_image, extra_image])
            log_prediction = np.concatenate([
                log_b0, log_b0 - bvalues * image
            ]) * roi
            prediction = np.exp(log_prediction) * roi
            image_nii = nibabel.load(find_file(config['image'], p_path))
            image_path = os.path.join(p_path, 'out-' + mask_name)
            prediction_nii = nibabel.Nifti1Image(
                np.moveaxis(log_prediction, 0, -1),
                image_nii.get_qform(), image_nii.header
            )
            prediction_nii.to_filename(image_path)
        else:
            lr_image = np.expand_dims(lr_image, axis=1)
            prediction = net.patch_inference(
                lr_image, config['test_patch'], config['test_batch'],
                directions[:21, ...], sub_i, len(testing_subjects),
                test_start
            ) * roi

        image_nii = nibabel.load(find_file(config['image'], p_path))
        prediction_nii = nibabel.Nifti1Image(
            np.moveaxis(prediction, 0, -1),
            image_nii.get_qform(), image_nii.header
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
    models = importlib.import_module('models')
    network_class = getattr(models, config['network'])
    try:
        encoder_filters = config['encoder_filters']
    except KeyError:
        encoder_filters = None
    try:
        decoder_filters = config['decoder_filters']
    except KeyError:
        decoder_filters = None
    try:
        heads = config['heads']
    except KeyError:
        heads = 32
    model_base = os.path.splitext(os.path.basename(options['config']))[0]
    print(
        '{:}[{:}] {:}<Diffusion super-resolution framework>{:}'.format(
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
            '{:}[{:}] {:}Starting cross-validation (model: {:}){:}'
            ' (seed {:d}){:}'.format(
                c['clr'] + c['c'], strftime("%H:%M:%S"), c['g'],
                c['b'] + model_base + c['nc'] + c['g'],
                c['nc'] + c['y'], seed, c['nc']
            )
        )
        # Network init (random weights)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        net = network_class(
            encoder_filters=encoder_filters,
            decoder_filters=decoder_filters,
            heads=heads
        )
        starting_model = os.path.join(
            model_path, '{:}-start.s{:05d}.pt'.format(model_base, seed),
        )
        net.save_model(starting_model)
        n_param = sum(
            p.numel() for p in net.parameters() if p.requires_grad
        )

        print(
            '{:}Training with seed {:} - {:02d}/{:02d} '
            '({:} parameters)'.format(
                c['clr'] + c['c'], c['b'] + str(seed) + c['nc'],
                test_n + 1, len(config['seeds']),
                c['b'] + str(n_param) + c['nc']
            )
        )

        random.shuffle(subjects)
        n_test = int(math.ceil(len(subjects) / n_folds))
        # Cross-validation loop
        for i in range(n_folds):
            net = network_class(
                encoder_filters=encoder_filters,
                decoder_filters=decoder_filters,
                heads=heads
            )
            net.load_model(starting_model)
            testing = subjects[i * n_test:(i + 1) * n_test]
            not_testing = subjects[(i + 1) * n_test:] + subjects[0:i * n_test]
            if val_split > 0:
                validation = not_testing[:len(not_testing) * val_split]
                training = not_testing[len(not_testing) * val_split:]
            else:
                training = validation = not_testing

            fold_name = os.path.join(
                model_path,
                '{:}.n{:02d}.s{:05d}.pt'.format(model_base, i, seed)
            )
            train(config, net, training, validation, fold_name, 2)
            test(config, seed, net, model_base, testing, 2)


if __name__ == '__main__':
    main()
