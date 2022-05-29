import numpy as np
import scipy.ndimage as ndi
import torch
import pdb

def bilateralFtr1D(yi, sSpatial=3, sIntensity=1):

    res =torch.zeros_like(yi).cuda()
    F_rec =torch.zeros_like(yi).cuda()
    radius = int(np.floor(2 * sSpatial))
    filterSize = int((2 * radius) + 1)
    expS0 = torch.zeros(filterSize).cuda()
    ###calculate  expS
    for jj in range(filterSize):
        expS0[jj]=torch.tensor(np.exp(-(radius-jj)*(radius-jj)/radius/radius)).cuda()
    for ii in range(yi.shape[0]):
        y = yi[ii]


        # 1d data dimensions
        width = y.shape[0]

        # 1d resulting data
        ret = torch.zeros(width).cuda()

        for i in range(width):
            ## To prevent accessing values outside of the array
            # The left part of the lookup area, clamped to the boundary
            xmin = max(i - radius, 0);
            # How many columns were outside the image, on the left?
            dxmin = xmin - (i - radius);

            # The right part of the lookup area, clamped to the boundary
            xmax = min(i + radius+1, width);


            # The left expression in the bilateral filter equation
            # We take only the relevant parts of the matrix of the
            # Gaussian weights - we use dxmin, dxmax, dymin, dymax to
            # ignore the parts that are outside the image

            # The right expression in the bilateral filter equation
            dy = y[xmin:xmax] - y[i]
            dIsquare = (dy * dy)
            expI = torch.exp(- dIsquare / (sIntensity * sIntensity))

            # The bilater filter (weights matrix)
            if i<=radius:
                expS = expS0[radius-i:2*radius+1]

            else:
                if 750-i<=radius:
                    expS = expS0[0:xmax-xmin]

                else:
                    expS =expS0

            F = expI * expS

            # Normalized bilateral filter
            Fnormalized = F / sum(F)

            # Multiply the area by the filter
            tempY = y[xmin:xmax] * Fnormalized

            # The resulting pixel is the sum of all the pixels in
            # the area, according to the weights of the filter
            # ret(i,j,R) = sum (tempR(:))
            ret[i] = sum(tempY)
            F_rec[ii,i]=sum(F)
        res[ii] = ret

    return res,F_rec
