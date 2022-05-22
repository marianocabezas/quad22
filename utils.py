import time
import os
import re
import sys
import traceback
from functools import reduce
from scipy import ndimage as nd
import numpy as np
from nibabel import load as load_nii
from scipy.ndimage.morphology import binary_dilation as imdilate
from scipy.ndimage.morphology import binary_erosion as imerode
import torch


"""
Utility functions
"""


def color_codes():
    """
    Function that returns a custom dictionary with ASCII codes related to
    colors.
    :return: Custom dictionary with ASCII codes for terminal colors.
    """
    codes = {
        'nc': '\033[0m',
        'b': '\033[1m',
        'k': '\033[0m',
        '0.25': '\033[30m',
        'dgy': '\033[30m',
        'r': '\033[31m',
        'g': '\033[32m',
        'gc': '\033[32m;0m',
        'bg': '\033[32;1m',
        'y': '\033[33m',
        'c': '\033[36m',
        '0.75': '\033[37m',
        'lgy': '\033[37m',
        'clr': '\033[K',
    }
    return codes


def find_file(name, dirname):
    """

    :param name:
    :param dirname:
    :return:
    """
    result = list(filter(
        lambda x: not os.path.isdir(x) and re.search(name, x),
        os.listdir(dirname)
    ))

    return os.path.join(dirname, result[0]) if result else None


def get_dirs(path):
    """
    Function to get the folder name of the patients given a path.
    :param path: Folder where the patients should be located.
    :return: List of patient names.
    """
    # All patients (full path)
    patient_paths = sorted(
        filter(
            lambda d: os.path.isdir(os.path.join(path, d)),
            os.listdir(path)
        )
    )
    # Patients used during training
    return patient_paths


def print_message(message):
    """
    Function to print a message with a custom specification
    :param message: Message to be printed.
    :return: None.
    """
    c = color_codes()
    dashes = ''.join(['-'] * (len(message) + 11))
    print(dashes)
    print(
        '%s[%s]%s %s' %
        (c['c'], time.strftime("%H:%M:%S", time.localtime()), c['nc'], message)
    )
    print(dashes)


def time_f(f, stdout=None, stderr=None):
    """
    Function to time another function.
    :param f: Function to be run. If the function has any parameters, it should
    be passed using the lambda keyword.
    :param stdout: File where the stdout will be redirected. By default we use
    the system's stdout.
    :param stderr: File where the stderr will be redirected. By default we use
    the system's stderr.
    :return: The result of running f.
    """
    # Init
    stdout_copy = sys.stdout
    if stdout is not None:
        sys.stdout = stdout

    start_t = time.time()
    try:
        ret = f()
    except Exception as e:
        ret = None
        exc_type, exc_value, exc_traceback = sys.exc_info()
        print('{0}: {1}'.format(type(e).__name__, e), file=stderr)
        traceback.print_tb(exc_traceback, file=stderr)
    finally:
        if stdout is not None:
            sys.stdout = stdout_copy

    print(
        time.strftime(
            'Time elapsed = %H hours %M minutes %S seconds',
            time.gmtime(time.time() - start_t)
        )
    )
    return ret


def time_to_string(time_val):
    """
    Function to convert from a time number to a printable string that
     represents time in hours minutes and seconds.
    :param time_val: Time value in seconds (using functions from the time
     package)
    :return: String with a human format for time
    """

    if time_val < 60:
        time_s = '%ds' % time_val
    elif time_val < 3600:
        time_s = '%dm %ds' % (time_val // 60, time_val % 60)
    else:
        time_s = '%dh %dm %ds' % (
            time_val // 3600,
            (time_val % 3600) // 60,
            time_val % 60
        )
    return time_s


def get_int(string):
    """
    Function to get the int number contained in a string. If there are more
    than one int number (or there is a floating point number), this function
    will concatenate all digits and return an int, anyways.
    :param string: String that contains an int number
    :return: int number
    """
    return int(''.join(filter(str.isdigit, string)))


"""
Data related functions
"""


def get_bb(mask, dilate=0):
    """

    :param mask:
    :param dilate:
    :return:
    """
    if dilate > 0:
        mask = imdilate(mask, iterations=dilate)
    idx = np.where(mask)
    bb = tuple(
        slice(min_i, max_i)
        for min_i, max_i in zip(
            np.min(idx, axis=-1), np.max(idx, axis=-1)
        )
    )
    return bb


def get_mask(mask_name, dilate=0, dtype=np.uint8):
    """
    Function to load a mask image
    :param mask_name: Path to the mask image file
    :param dilate: Dilation radius
    :param dtype: Data type for the final mask
    :return:
    """
    # Lesion mask
    mask_image = (load_nii(mask_name).get_fdata() > 0.5).astype(dtype)
    if dilate > 0:
        mask_d = imdilate(
            mask_image,
            iterations=dilate
        )
        mask_e = imerode(
            mask_image,
            iterations=dilate
        )
        mask_image = np.logical_and(mask_d, np.logical_not(mask_e)).astype(dtype)

    return mask_image


def get_normalised_image(
    image, mask=None, dtype=np.float32, masked=False
):
    """
    Function to a load an image and normalised it (0 mean / 1 standard
     deviation)
    :param image: Image to be normalised
    :param mask: Mask defining the region of interest
    :param dtype: Data type for the final image
    :param masked: Whether to mask the image or not
    :return:
    """

    # If no mask is provided we use the image as a mask (all non-zero values)
    if mask is None:
        mask_bin = image.astype(np.bool)
    else:
        mask_bin = mask.astype(np.bool)

    if len(image.shape) > len(mask_bin.shape):
        image_list = []
        for i in range(image.shape[-1]):
            image_i = image[i, ...]
            image_mu = np.mean(image_i[mask_bin])
            image_sigma = np.std(image_i[mask_bin])
            if masked:
                image_i = image_i * mask_bin.astype(image.dtype)
            norm_image = ((image_i - image_mu) / image_sigma).astype(dtype)
            image_list.append(norm_image)
        output = np.stack(image_list, axis=0)

    else:
        # Parameter estimation using the mask provided
        image_mu = np.mean(image[mask_bin])
        image_sigma = np.std(image[mask_bin])
        if masked:
            image = image * mask_bin.astype(image.dtype)

        output = ((image - image_mu) / image_sigma).astype(dtype)

    return output


def remove_small_regions(img_vol, min_size=3):
    """
        Function that removes blobs with a size smaller than a minimum from a mask
        volume.
        :param img_vol: Mask volume. It should be a numpy array of type bool.
        :param min_size: Minimum size for the blobs.
        :return: New mask without the small blobs.
    """
    blobs, _ = nd.measurements.label(
        img_vol,
        nd.morphology.generate_binary_structure(3, 3)
    )
    labels = list(filter(bool, np.unique(blobs)))
    areas = [np.count_nonzero(np.equal(blobs, lab)) for lab in labels]
    nu_labels = [lab for lab, a in zip(labels, areas) if a >= min_size]
    nu_mask = np.isin(blobs, nu_labels)
    return nu_mask


def remove_boundary_regions(img_vol, roi, thickness=1):
    """
        Function that removes blobs with a size smaller than a minimum from a
        mask volume.
        :param img_vol: Mask volume. It should be a numpy array of type bool.
        :param roi: Region of interest mask. It should be a numpy array of type
         bool.
        :param thickness: Thickness of the boundary ribbon.
        :return: New mask without the small blobs.
    """

    small_roi = imerode(roi, iterations=thickness)
    boundary = np.logical_and(roi, np.logical_not(small_roi))
    blobs, _ = nd.measurements.label(
        img_vol,
        nd.morphology.generate_binary_structure(3, 3)
    )
    boundary_labels = list(np.unique(blobs[boundary]))
    nu_mask = np.isin(blobs, boundary_labels, invert=True)
    return nu_mask


def to_torch_var(
    np_array,
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    requires_grad=False,
    dtype=torch.float32
):
    """
    Function to convert a numpy array into a torch tensor for a given device
    :param np_array: Original numpy array
    :param device: Device where the tensor will be loaded
    :param requires_grad: Whether it requires autograd or not
    :param dtype: Datatype for the tensor
    :return:
    """
    var = torch.tensor(
        np_array,
        requires_grad=requires_grad,
        device=device,
        dtype=dtype
    )
    return var
