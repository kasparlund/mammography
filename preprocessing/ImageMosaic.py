from __future__ import division
import numpy as np


class MosaicException(Exception):
    pass


def create_mosaic(
        images, nrows=None, ncols=None, border_val=0, border_size=5,
        rows_first=True):
    """
    Creates a mosaic of input images. 'images' should be an iterable of
    2D or 3D images. If they're 3D then they must have the shape
    (spatial, spatial, channels).

    Creates a square mosaic unless nrows and/or ncols is not 'None'. If both
    are not 'None' then ncols * nrows >= len(images)

    :param images: An iterable of 2D or 3D numpy arrays
    :param nrows: Custom number of rows
    :param ncols: Custom number of columns
    :param border_val: The value of the background and borders
    :param border_size: The size of the border between images
    :param rows_first: A boolean indicating if the mosaic should be populated
    by rows (True) or columns (False) first
    :return: 
    """

    # Validate
    if border_size < 0:
        raise MosaicException("'border_size' must be >= 0")

    # Trivial cases
    if len(images) == 0:
        return None

    if len(images) == 1:
        return images[0]

    imshapes = [image.shape for image in images]
    imdims = [len(imshape) for imshape in imshapes]
    dtypes = [image.dtype for image in images]

    # Images must be all 2D or 3D
    if not set(imdims).issubset({2, 3}):
        raise MosaicException("All images must be 2D or 3D numpy arrays")

    # Images must all have the same shape
    if len(set(imdims)) != 1:
        raise MosaicException("Images cannot be mixed 2D and 3D")

    # If images are 3D then they must all have the same number of channels
    if imdims[0] == 3:
        num_channels = [imshape[2] for imshape in imshapes]
        if len(set(num_channels)) != 1:
            raise MosaicException(
                "3D images must have the same number of channels")

    # Images must all have the same dtype
    if len(set(dtypes)) != 1:
        raise MosaicException("All images must have the same dtype")

    # Embed images if they're differently sized
    if len(set(imshapes)) != 1:
        new_images = []
        max_dims = [
            max([imshape[i] for imshape in imshapes])
            for i in range(imdims[0])]
        for image in images:
            diff = [max_dims[i] - image.shape[i] for i in range(len(max_dims))]
            embed_box = border_val * np.ones(max_dims)
            start0 = diff[0] // 2
            end0 = start0 + image.shape[0]
            start1 = diff[1] // 2
            end1 = start1 + image.shape[1]
            embed_box[start0:end0, start1:end1] = image
            new_images.append(embed_box)
        images = new_images

    # Set up grid
    if ncols is None and nrows is None:
        n_cols = int(np.ceil(np.sqrt(len(images))))
        n_rows = int(np.ceil(len(images) / n_cols))
    elif ncols is None and nrows is not None:
        n_rows = nrows
        n_cols = int(np.ceil(len(images) / n_rows))
    elif ncols is not None and nrows is None:
        n_cols = ncols
        n_rows = int(np.ceil(len(images) / n_cols))
    else:
        n_cols = ncols
        n_rows = nrows

    if n_rows <= 0 or n_cols <= 0:
        raise MosaicException("'nrows' and 'ncols' must both be > 0")

    if n_cols * n_rows < len(images):
        raise MosaicException(
            "Mosaic grid too small: n_rows * n_cols < len(images)")

    size_0 = images[0].shape[0]
    size_1 = images[0].shape[1]

    if imdims[0] == 2:
        out_block = border_val * np.ones(
            (size_0 * n_rows + border_size * (n_rows - 1),
             size_1 * n_cols + border_size * (n_cols - 1)),
            dtype=dtypes[0])
    else:
        out_block = border_val * np.ones(
            (size_0 * n_rows + border_size * (n_rows - 1),
             size_1 * n_cols + border_size * (n_cols - 1),
             images[0].shape[2]), dtype=dtypes[0])
    for fld in range(len(images)):
        if rows_first:
            index_0 = int(np.floor(fld / n_cols))
            index_1 = fld - (n_cols * index_0)
        else:
            index_1 = int(np.floor(fld / n_rows))
            index_0 = fld - (n_rows * index_1)

        start0 = index_0 * (size_0 + border_size)
        end0 = start0 + size_0
        start1 = index_1 * (size_1 + border_size)
        end1 = start1 + size_1
        out_block[start0:end0, start1:end1] = images[fld]

    return out_block
