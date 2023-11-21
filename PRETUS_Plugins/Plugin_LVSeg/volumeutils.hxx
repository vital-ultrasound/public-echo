#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <ifindImage.h>

namespace volume {


template <class T>
static int compute_LVV_simpsons(const typename T::Pointer im){
    int nlayers = im->GetNumberOfLayers();
    auto segmentation = im->GetOverlay(nlayers-1);
//    endo = (seg > 128 ).astype(float) # seg is 0 to 255
//     X, Y = np.nonzero(endo)
//     dim = X.shape
//     dim = int(dim[0])
//     cord = np.zeros((dim, 2))
//     X = X.astype(np.float64)
//     Y = Y.astype(np.float64)
//     cord[:, 0] = X
//     cord[:, 1] = Y
//     pca = PCA(n_components=2)
//     pca.fit(cord)
//     X_pca = pca.transform(cord)

//     /// order the points
//     sorted_idx = np.argsort(X_pca[:, 0])

//     X_axis = pca.inverse_transform(X_pca)
////     X_new = np.zeros((dim,))
////     Y_new = np.zeros((dim,))
//     X_new = X_axis[:, 0]
//     Y_new = X_axis[:, 1]

////     _mid_valve_pca = [X_pca[sorted_idx[0], 0], X_pca[sorted_idx[0], 1]]
////     _apex_pca = [X_pca[sorted_idx[-1], 0], X_pca[sorted_idx[-1], 1]]
////     length_LV_pca = distance.pdist([_mid_valve_pca, _apex_pca])[0]

//     _mid_valve = [X_new[sorted_idx[-1]], Y_new[sorted_idx[-1]]]
//     _apex = [X_new[sorted_idx[0]], Y_new[sorted_idx[0]]]

//     /// Compute simpsons rule
//     length_LV = distance.pdist([_mid_valve, _apex])[0]

//     /// now I make the disks
//     /// v = np.array(_mid_valve) - np.array(_apex)
//     /// v = v/length_LV

//     disk_thickness = length_LV / N
//     x0_pca = X_pca[sorted_idx[0], 0]
//     disk_vols = []
//     for n in range(N):
//         xmin = x0_pca + disk_thickness * n
//         xmax = x0_pca + disk_thickness * (n+1)

//         /// find the width at this x
//         ys = X_pca[(X_pca[:, 0] >= xmin) & (X_pca[:, 0] < xmax), 1]
//         if len(ys) <= 0:
//             continue

//         r = (ys.max() - ys.min())/2
//         disk_vol = np.pi * r**2 * disk_thickness
//         disk_vols.append(disk_vol)
//     LV_vol_mm3 = np.sum(disk_vols) * np.mean(spacing)**3 /// assuming isotropic

    return 0;

}



}
