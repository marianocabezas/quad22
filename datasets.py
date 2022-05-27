import itertools
from copy import deepcopy
import numpy as np
from torch.utils.data.dataset import Dataset


''' Utility function for patch creation '''


def tokenize(dmri, directions, bvalues):
    hr_dmri = np.expand_dims(dmri, axis=1)
    hr_dir = np.broadcast_to(
        np.expand_dims(directions, axis=(2, 3, 4)),
        directions.shape + hr_dmri.shape[2:]
    )
    hr_bvalues = np.broadcast_to(
        np.expand_dims(bvalues, axis=(1, 2, 3, 4)),
        bvalues.shape + hr_dmri.shape[1:]
    )
    hr_data = np.concatenate([hr_bvalues, hr_dir, hr_dmri], axis=1)

    return hr_data


def centers_to_slice(voxels, patch_half):
    """
    Function to convert a list of indices defining the center of a patch, to
    a real patch defined using slice objects for each dimension.
    :param voxels: List of indices to the center of the slice.
    :param patch_half: List of integer halves (//) of the patch_size.
    """
    slices = [
        center_to_slice(voxel, patch_half) for voxel in voxels
    ]
    return slices


def center_to_slice(voxel, patch_half):
    slice_i = tuple(
        slice(idx - p_len, idx + p_len)
        for idx, p_len in zip(voxel, patch_half)
    )

    return slice_i


def get_slices(masks, patch_size, overlap):
    """
    Function to get all the patches with a given patch size and overlap between
    consecutive patches from a given list of masks. We will only take patches
    inside the bounding box of the mask. We could probably just pass the shape
    because the masks should already be the bounding box.
    :param masks: List of masks.
    :param patch_size: Size of the patches.
    :param overlap: Overlap on each dimension between consecutive patches.

    """
    # Init
    # We will compute some intermediate stuff for later.
    patch_half = [p_length // 2 for p_length in patch_size]
    steps = [max(p_length - o, 1) for p_length, o in zip(patch_size, overlap)]

    # We will need to define the min and max pixel indices. We define the
    # centers for each patch, so the min and max should be defined by the
    # patch halves.
    min_bb = [patch_half] * len(masks)
    max_bb = [
        [
            max_i - p_len for max_i, p_len in zip(mask.shape, patch_half)
        ] for mask in masks
    ]

    # This is just a "pythonic" but complex way of defining all possible
    # indices given a min, max and step values for each dimension.
    dim_ranges = [
        map(
            lambda t: np.concatenate([np.arange(*t), [t[1]]]),
            zip(min_bb_i, max_bb_i, steps)
        ) for min_bb_i, max_bb_i in zip(min_bb, max_bb)
    ]

    # And this is another "pythonic" but not so intuitive way of computing
    # all possible triplets of center voxel indices given the previous
    # indices. I also added the slice computation (which makes the last step
    # of defining the patches).
    patch_slices = [
        centers_to_slice(
            itertools.product(*dim_range), patch_half
        ) for dim_range in dim_ranges
    ]

    return patch_slices


def get_centers(masks, patch_size, overlap):
    """
    Function to get all the patches with a given patch size and overlap between
    consecutive patches from a given list of masks. We will only take patches
    inside the bounding box of the mask. We could probably just pass the shape
    because the masks should already be the bounding box.
    :param masks: List of masks.
    :param patch_size: Size of the patches.
    :param overlap: Overlap on each dimension between consecutive patches.

    """
    # Init
    # We will compute some intermediate stuff for later.
    patch_half = [p_length // 2 for p_length in patch_size]
    steps = [max(p_length - o, 1) for p_length, o in zip(patch_size, overlap)]

    # We will need to define the min and max pixel indices. We define the
    # centers for each patch, so the min and max should be defined by the
    # patch halves.
    min_bb = [patch_half] * len(masks)
    max_bb = [
        [
            max_i - p_len for max_i, p_len in zip(mask.shape, patch_half)
        ] for mask in masks
    ]

    # This is just a "pythonic" but complex way of defining all possible
    # indices given a min, max and step values for each dimension.
    dim_ranges = [
        map(
            lambda t: np.concatenate([np.arange(*t), [t[1]]]),
            zip(min_bb_i, max_bb_i, steps)
        ) for min_bb_i, max_bb_i in zip(min_bb, max_bb)
    ]

    return [list(itertools.product(*dim_range)) for dim_range in dim_ranges]


def randomized_shift(center, size, patch_size, max_shift):
    shift = tuple(
        np.random.randint(-max_shift_i, max_shift_i, 1)
        for max_shift_i in max_shift
    )
    new_center = tuple(
        int(min(max(c + sh, p // 2), s - p // 2))
        for c, sh, s, p in zip(center, shift, size, patch_size)
    )

    return new_center


''' Datasets '''


class DiffusionDataset(Dataset):
    def __init__(
        self, dmri, rois, directions, bvalues, patch_size=32,
        overlap=0, min_lr=22, max_lr=22, shift=True
    ):
        # Init
        if type(patch_size) is not tuple:
            self.patch_size = (patch_size,) * 3
        else:
            self.patch_size = patch_size
        self.patch_half = tuple(ps // 2 for ps in self.patch_size)
        if type(overlap) is not tuple:
            self.overlap = (overlap,) * 3
        else:
            self.overlap = overlap
        self.shift = shift

        self.images = dmri
        self.rois = rois
        self.directions = directions
        self.bvalues = bvalues
        n_directions = [len(bvalue) > 7 for bvalue in self.bvalues]
        assert np.all(n_directions), 'The inputs are already low resolution'
        if min_lr < 7:
            self.min_lr = 7
        else:
            self.min_lr = min_lr
        if max_lr < self.min_lr:
            self.max_lr = self.min_lr
        else:
            self.max_lr = max_lr

        # We get the preliminary patch slices (inside the bounding box)...
        slices = get_centers(self.rois, self.patch_size, self.overlap)

        # ... however, being inside the bounding box doesn't guarantee that the
        # patch itself will contain any lesion voxels. Since, the lesion class
        # is extremely underrepresented, we will filter this preliminary slices
        # to guarantee that we only keep the ones that contain at least one
        # lesion voxel.
        self.patch_centers = [
            (s, i) for i, s_i in enumerate(slices) for s in s_i
        ]

    def __getitem__(self, index):
        center_i, case_idx = self.patch_centers[index]
        dmri = self.images[case_idx]

        none_slice = (slice(None),)
        if self.shift:
            shifted_center_i = randomized_shift(
                center_i, dmri.shape[1:], self.patch_size, self.patch_half
            )
            slice_i = center_to_slice(shifted_center_i, self.patch_half)
        else:
            slice_i = center_to_slice(center_i, self.patch_half)

        patch = dmri[none_slice + slice_i].astype(np.float32)
        print(patch.shape)
        dirs = self.directions[case_idx].astype(np.float32)
        bvalues = self.bvalues[case_idx].astype(np.float32)
        if self.min_lr == self.max_lr:
            lr_end = self.min_lr
        else:
            lr_end = np.random.randint(self.min_lr, self.max_lr, 1)
        hr_data = tokenize(patch, dirs, bvalues)
        key_data = deepcopy(hr_data[:lr_end, ...])
        query_data = deepcopy(hr_data[lr_end:, :-1, ...])
        target_data = hr_data[lr_end:, -1, ...]

        return (key_data, query_data), target_data

    def __len__(self):
        return len(self.patch_centers)
