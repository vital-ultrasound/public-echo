"""
utilities like get beam source, convert to cartesian/spherical, etc.
"""
# Author: Alberto Gomez <alberto.gomez@kcl.ac.uk>
# 	  King's College London, UK

import SimpleITK as sitk
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn


def displayImage(images, mode='sidewise', labels=None, special={'diff': None}, slice = 0.5 ):
    """Display one or more sitk images
    Arguments:
        images -- tuple with ine or more images
        labels -- tuple of strings with a label for each imge
        special -- special representations. Examples:
            special = {'diff' : (1, 0), # adds the difference between image 1 and image 0
            }
        slice -- slice number, in percent of total
        """

    datas = []
    for i in range(len(images)):
        data = sitk.GetArrayFromImage(images[i]).transpose()
        datas.append(data)

    if 'diff' in special and special['diff'] is not None:
        diff_data = datas[special['diff'][0]]-datas[special['diff'][1]]
        datas.append(diff_data)
        labels+=('difference',)

    for i in range(len(datas)):
        plt.subplot(1, len(datas), i+1)

        slicen = int(data.shape[-1]*slice)
        plt.imshow(datas[i][...,slicen].squeeze(), cmap='gray')
        if labels is not None:
            plt.title(labels[i])
    plt.show()


def sitkToTorch(sitk_image, transpose = True):
    np_image = sitk.GetArrayFromImage(sitk_image)  # need to permute indices for consistency
    if transpose is True:
        np_image = np.transpose(np_image)
        tensor_info = {'spacing': sitk_image.GetSpacing(), 'origin': sitk_image.GetOrigin(),
                       'size': sitk_image.GetSize()}
    else:
        tensor_info = {'spacing': sitk_image.GetSpacing()[::-1], 'origin': sitk_image.GetOrigin()[::-1],
                       'size': sitk_image.GetSize()[::-1]}

    image_tensor = torch.from_numpy(np_image)
    return image_tensor, tensor_info

def torchToSitk(torch_tensor, tensor_info, transpose = True):
    """

    Args:
        torch_tensor: torch tensor (cpu)
        tensor_info: dictionary
        transpose: if data was transposed at load time or not

    Returns:

    """
    np_image = torch_tensor.numpy()
    if transpose is True:
        np_image = np.transpose(np_image)
        # if data was transposed at load time, then spacing/origin were not
        spacing = tensor_info['spacing']
        origin = tensor_info['origin']
    else:
        spacing = tensor_info['spacing'][::-1]
        origin = tensor_info['origin'][::-1]

    sitk_image = sitk.GetImageFromArray(np_image)
    sitk_image.SetSpacing(spacing)
    sitk_image.SetOrigin(origin)
    return sitk_image

def GetInfo(itk_img):
    info = {'spacing': None, 'origin': None, 'size': None}
    info['spacing'] = list(itk_img.GetSpacing())
    info['origin'] = list(itk_img.GetOrigin())
    info['size'] = list(itk_img.GetSize())
    return info



def crop_image(image, bounds=None, threshold = 0, extra_crop = 0):
    """Crop an image to a specified bound or to nonzero values"""

    D = image.GetDimension()

    if bounds is None:
        # get bounds from values above the threshold
        image_np = sitk.GetArrayFromImage(image)
        image_np_binary = image_np>threshold
        sum_x = np.sum(image_np_binary, axis=1)
        sum_y = np.sum(image_np_binary, axis=0)
        if D>2:
            # change the names of the arrays to reflect the dim
            sum_z = np.sum(sum_x, axis=1)
            sum_y = np.sum(sum_y, axis=1)
            sum_x = np.sum(np.sum(image_np_binary, axis=0), axis = 0)

        # find the first and last nonzeros
        ix0 = (sum_x != 0).argmax(axis=0)
        ix1 = sum_x.shape[0]-(sum_x[::-1] != 0).argmax(axis=0)
        iy0 = (sum_y != 0).argmax(axis=0)
        iy1 = sum_y.shape[0] - (sum_y[::-1] != 0).argmax(axis=0)

        lowerBoundary = (ix0, iy0)
        upperBoundary = (ix1, iy1)

        if D > 2:
            iz0 = (sum_z != 0).argmax(axis=0)
            iz1 = sum_z.shape[0] - (sum_z[::-1] != 0).argmax(axis=0)
            lowerBoundary += (iz0, )
            upperBoundary += (iz1, )

    else:
        lowerBoundary = bounds[0:2:]
        upperBoundary = bounds[1:2:]

    if D==2:
        output = image[lowerBoundary[0]+extra_crop:upperBoundary[0]-extra_crop, lowerBoundary[1]+extra_crop:upperBoundary[1]-extra_crop]
    else:
        output = image[lowerBoundary[0]+extra_crop:upperBoundary[0]-extra_crop, lowerBoundary[1]+extra_crop:upperBoundary[1]-extra_crop,
                 lowerBoundary[2]+extra_crop:upperBoundary[2]-extra_crop]

    return output


