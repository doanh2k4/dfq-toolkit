"""
This module provides utilities for display images in Jupyter notebooks.
"""

import torch
import torchvision


def denormalize(img):
    """Convert ImageNet scale to regular scale

    Parameters
    ----------
    img : Tensor

    Returns
    -------
    Tensor
    """
    invTrans = torchvision.transforms.Compose(
        [
            torchvision.transforms.Normalize(
                mean=[0.0, 0.0, 0.0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
            ),
            torchvision.transforms.Normalize(
                mean=[-0.485, -0.456, -0.406], std=[1.0, 1.0, 1.0]
            ),
        ]
    )

    inv_tensor = invTrans(img)
    return torchvision.transforms.ToPILImage()(inv_tensor)


def showone(ckpt_path, normalize):
    """Display the second picture in the batch of `ckpt_path`.

    Parameters
    ----------
    ckpt_path : str
    normalize : bool
        True -> the original ckpt is ImageNet scaled.
        False -> the original ckpt is [0, 1] scaled.

    NOTE: For YOLO models, normalize should always be False because it is [0, 1] scaled.

    Returns
    -------
    A PILImage
    """

    ckpt = torch.load(ckpt_path, map_location="cpu")
    img = ckpt[1]
    if normalize:
        return denormalize(img)
    return torchvision.transforms.ToPILImage()(img)
