
import torch
import unet as lvunet
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from scipy.spatial import distance

import warnings
# need to do this because scipy issues this warning. An alternative
# would be to upgrade scipy but this would require python > 3.7
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

net = None
threshold = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ---------------- Volumetrics code ----------------------

def compute_volume_Simpsons(seg, spacing, N = 20):
    endo = (seg > 128 ).astype(float) # seg is 0 to 255
    X, Y = np.nonzero(endo)
    dim = X.shape
    dim = int(dim[0])
    cord = np.zeros((dim, 2), dtype=np.float32)
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)
    cord[:, 0] = X
    cord[:, 1] = Y
    pca = PCA(n_components=2)
    pca.fit(cord) # here is where the visible deprecation warning is issued
    X_pca = pca.transform(cord)
    # order the points
    sorted_idx = np.argsort(X_pca[:, 0])

    X_axis = pca.inverse_transform(X_pca)
    # X_new = np.zeros((dim,))
    # Y_new = np.zeros((dim,))
    X_new = X_axis[:, 0]
    Y_new = X_axis[:, 1]

    # _mid_valve_pca = [X_pca[sorted_idx[0], 0], X_pca[sorted_idx[0], 1]]
    # _apex_pca = [X_pca[sorted_idx[-1], 0], X_pca[sorted_idx[-1], 1]]
    # length_LV_pca = distance.pdist([_mid_valve_pca, _apex_pca])[0]

    _mid_valve = [X_new[sorted_idx[-1]], Y_new[sorted_idx[-1]]]
    _apex = [X_new[sorted_idx[0]], Y_new[sorted_idx[0]]]

    # Compute simpsons rule
    length_LV = distance.pdist([_mid_valve, _apex])[0]

    # now I make the disks
    # v = np.array(_mid_valve) - np.array(_apex)
    # v = v/length_LV

    disk_thickness = length_LV / N
    x0_pca = X_pca[sorted_idx[0], 0]
    disk_vols = []
    for n in range(N):
        xmin = x0_pca + disk_thickness * n
        xmax = x0_pca + disk_thickness * (n+1)

        # find the width at this x
        ys = X_pca[(X_pca[:, 0] >= xmin) & (X_pca[:, 0] < xmax), 1]
        if len(ys) <= 0:
            continue

        r = (ys.max() - ys.min())/2
        disk_vol = np.pi * r**2 * disk_thickness
        disk_vols.append(disk_vol)
    LV_vol_mm3 = np.sum(disk_vols) * np.mean(spacing)**3 # assuming isotropic
    #
    #
    # import matplotlib.pyplot as plt
    # plt.imshow(seg.transpose())
    # plt.plot(X_new, Y_new, '.')
    # plt.plot(_apex[0], _apex[1], '*')
    # plt.plot(_mid_valve[0], _mid_valve[1], 'o')
    # plt.axis('equal')
    # plt.show()
    #
    # area_LV = np.sum(endo) # pixels square
    # print('Area LV: {}'.format(area_LV))
    # print('Length LV: {}'.format(length_LV))
    # LV = 0.85 * area_LV * area_LV / length_LV / 100 * np.prod(spacing)
    LV_vol_ml  =LV_vol_mm3
    return LV_vol_ml/1000.0


def compute_volume_SR(seg, spacing):
    endo = (seg > 128 ).astype(float) # seg is 0 to 255
    X, Y = np.nonzero(endo)
    dim = X.shape
    dim = int(dim[0])
    cord = np.zeros((dim, 2))
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)
    cord[:, 0] = X
    cord[:, 1] = Y
    pca = PCA(n_components=1)
    pca.fit(cord)
    X_pca = pca.transform(cord)
    X_axis = pca.inverse_transform(X_pca)
    X_new = np.zeros((dim,))
    Y_new = np.zeros((dim,))
    X_new = X_axis[:, 0]
    Y_new = X_axis[:, 1]

    _mid_valve = [X_new[0], Y_new[0]]
    _apex = [X_new[-1], Y_new[-1]]

    # plt.figure()
    # plt.imshow(seg)
    # plt.plot(_apex[1], _apex[0], "mo")
    # plt.plot(_mid_valve[1], _mid_valve[0], "go")
    # plt.show()

    # Compute simpsons rule
    length_LV = distance.pdist([_mid_valve, _apex])[0]
    area_LV = np.sum(endo) # pixels square
    LV = 0.85 * area_LV * area_LV / length_LV / 1000 * np.prod(spacing)
    # Debug this:
    #    a = np.max(X_new) - np.min(X_new)
    #    print(a * spacing[0])
    #    LV =2/3 * np.pi * a**2 * length_LV* np.prod(spacing)
    #    print(LV)
    return LV

def segmentation_to_volume(seg, pixel_size):
    # volume = compute_volume_SR(seg, pixel_size) #11 to 35ml
    volume = compute_volume_Simpsons(seg, pixel_size) # 13 to 46ml
    return volume

# -------------------------------------------
def initialize(input_size, python_path, modelname, th, verbose=False):
    global device
    global net
    global threshold
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = lvunet.UNet(input_size=input_size, blur_input=True)
    net.to(device)
    net.eval()
    if verbose:
        print(net)
        print("the model is : " + modelname)

    net_params = torch.load('{}/{}'.format(python_path, modelname))
    net.load_state_dict(net_params)
    threshold = th
    return True


def get_sobel_kernel(k=3):
    # get range
    range = np.linspace(-(k // 2), k // 2, k)
    # compute a grid the numerator and the axis-distances
    x, y = np.meshgrid(range, range)
    sobel_2D_numerator = x
    sobel_2D_denominator = (x ** 2 + y ** 2)
    sobel_2D_denominator[:, k // 2] = 1  # avoid division by zero
    sobel_2D = sobel_2D_numerator / sobel_2D_denominator
    return sobel_2D


def find_background_value(im):
    bck = stats.mode(im.reshape((-1)))
    return bck[0][0]


def dowork(image_cpp, spacing):
    """
    spacing: 2 element tuple spx and spy
    """


    with torch.no_grad():
        #np.save('/home/ag09/data/VITAL/input.npy', image_cpp)
        #image_cpp = image_cpp.transpose() # maybe do in cpp?
        #im = Image.fromarray(image_cpp)
        #im.save("/home/ag09/data/VITAL/input.png")
        # little hack to ensure that black is black
        bck = find_background_value(image_cpp)
        image_cpp[image_cpp<=bck] = bck
        image_cpp = torch.from_numpy(image_cpp).type(torch.float).to(device).unsqueeze(0).unsqueeze(0)
        image_cpp -= bck
        image_cpp /= torch.max(image_cpp)


        try:
            # output must be int8
            segmentation_out = net(image_cpp)
            do_edges=False
            if do_edges:
                k_sobel = 3
                sobel_2D = get_sobel_kernel(k_sobel)
                sobel_filter_x = torch.nn.Conv2d(in_channels=1,
                                                out_channels=1,
                                                kernel_size=k_sobel,
                                                padding=k_sobel // 2,
                                                bias=False)
                sobel_filter_x.weight[:] = torch.from_numpy(sobel_2D)
                sobel_filter_x.to(device)


#                sobel_filter_y = nn.Conv2d(in_channels=1,
#                                                out_channels=1,
#                                                kernel_size=k_sobel,
#                                                padding=k_sobel // 2,
#                                                bias=False)
#                sobel_filter_y.weight[:] = torch.from_numpy(sobel_2D.T)
#                sobel_filter_y.to(device)
                segmentation_out[segmentation_out>0.5] = 1
                segmentation_out = sobel_filter_x(segmentation_out)

            segmentation = (segmentation_out.squeeze()*255.0).type(torch.uint8).cpu().numpy()

            # compute the volume
            # volume = segmentation_to_volume(segmentation, spacing)
            # compute just the area
            volume = np.count_nonzero(segmentation > threshold)

            #im = Image.fromarray(segmentation)
            #im.save("/home/ag09/data/VITAL/output.png")
            #np.save('/home/ag09/data/VITAL/output.npy', segmentation)
            return (segmentation, volume)
        except Exception as inst:
            print("LVSeg_worker.py ::WARNING::  exception")
            print(type(inst))    # the exception instance
            print(inst.args)     # arguments stored in .args
            print(inst)          # _
