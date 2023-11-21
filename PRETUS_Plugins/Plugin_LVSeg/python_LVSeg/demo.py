

import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
import LVSeg_worker as lvseg_worker
import SimpleITK as sitk
import time
import lvutils as utils
import matplotlib.pyplot as plt

def resample(image, desired_size):
    size = desired_size
    origin = image.GetOrigin()
    spacing = [(s2 - 1) * sp2 / (s1 - 1) for s1, s2, sp2 in zip(desired_size, image.GetSize(), image.GetSpacing())]

    ref = sitk.Image(size, sitk.sitkInt8)
    ref.SetOrigin(origin)
    ref.SetSpacing(spacing)
    ref.SetDirection(image.GetDirection())

    # resample
    identity_transform = sitk.AffineTransform(image.GetDimension())
    identity_transform.SetIdentity()
    image = sitk.Resample(image, ref, identity_transform, sitk.sitkLinear, 0)

    return image


def read_data(path, desired_size, crop=None):
    image = sitk.ReadImage(path)
    if len(image.GetSize()) > 2:
        if image.GetSize()[-1] > 1:
            idx_ = np.random.randint(0, image.GetSize()[-1])
            image = image[:, :, idx_]
        else:
            image = image[:, :, 0]
    if crop is not None:
        cb = [int(c * s) for c, s in zip(crop, image.GetSize() * 2)]
        image = image[cb[0]:(cb[0]+cb[2]), cb[1]:(cb[1]+cb[3])]
    image = resample(image, desired_size)
    image, info_im = utils.sitkToTorch(image, transpose=True)
    image = image.type(torch.float).numpy()

    return image

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Missing one argument-input image")
        exit(-1)
    im_file = sys.argv[1]
    print('Input image: {}'.format(im_file))

    modelfolder = 'model'
    modelname = 'model.pth'
    desired_size = (128, 128)
    rel_crop_bounds = (0.25, 0.2, 0.55, 0.65)
    X_test = read_data(path=im_file, desired_size=desired_size, crop=rel_crop_bounds)
    # apply transforms
    X_test = X_test.astype(np.uint8)
    imsize = X_test.shape

    lvseg_worker.initialize(imsize, modelfolder, modelname, th=0.5, verbose=True)
    imidx = 0

    # worker takes an image in [0, 1]

    print('Run the model inference')
    spacing = [1, 1]
    startt = time.time()
    pred_segmentation, _ = lvseg_worker.dowork(X_test.transpose(), spacing=spacing)
    endt = time.time()
    sitk.WriteImage(sitk.GetImageFromArray(X_test.transpose()), '/home/ag09/data/test/bmode_ds.mhd')
    sitk.WriteImage(sitk.GetImageFromArray(pred_segmentation), '/home/ag09/data/test/seg.mhd')
    print('Elapsed time: {}s'.format(endt - startt))
    if len(pred_segmentation.shape) > 2:
        pred_segmentation = pred_segmentation[0, ...].squeeze()


    # Perform a sanity check on some random test samples
    plt.imshow(X_test.transpose(), cmap='gray')
    plt.imshow(pred_segmentation > 200, alpha=pred_segmentation/255.0)
    plt.show()
