import numpy as np
import scipy.ndimage as ndi
def bilateralFtr1D(yi, sSpatial=5, sIntensity=1):
    res = np.zeros_like(yi)
    F_rec= np.zeros_like(yi)
    for ii in range(yi.shape[0]):
        y = yi[ii]
        F_array = np.zeros(y.shape[0])
        radius = int(np.floor(2 * sSpatial))
        filterSize = int((2 * radius) + 1)
        ftrArray = np.zeros(filterSize)
        ftrArray[radius] = 1

        # Compute the Gaussian filter part of the Bilateral filter
        gauss = ndi.gaussian_filter1d(ftrArray, sSpatial)

        # 1d data dimensions
        width = y.size

        # 1d resulting data
        ret = np.zeros(width)

        for i in range(width):
            ## To prevent accessing values outside of the array
            # The left part of the lookup area, clamped to the boundary
            xmin = max(i - radius, 1);
            # How many columns were outside the image, on the left?
            dxmin = xmin - (i - radius);

            # The right part of the lookup area, clamped to the boundary
            xmax = min(i + radius, width);
            # How many columns were outside the image, on the right?
            dxmax = (i + radius) - xmax;

            # The actual range of the array we will look at
            area = y[xmin:xmax]

            # The center position
            center = y[i]

            # The left expression in the bilateral filter equation
            # We take only the relevant parts of the matrix of the
            # Gaussian weights - we use dxmin, dxmax, dymin, dymax to
            # ignore the parts that are outside the image
            expS = gauss[(1 + dxmin):(filterSize - dxmax)]

            # The right expression in the bilateral filter equation
            dy = y[xmin:xmax] - y[i]
            dIsquare = (dy * dy)
            expI = np.exp(- dIsquare / (sIntensity * sIntensity))

            # The bilater filter (weights matrix)
            F = expI * expS

            # Normalized bilateral filter
            Fnormalized = F / sum(F)

            # Multiply the area by the filter
            tempY = y[xmin:xmax] * Fnormalized

            # The resulting pixel is the sum of all the pixels in
            # the area, according to the weights of the filter
            # ret(i,j,R) = sum (tempR(:))
            ret[i] = sum(tempY)
            F_array[i]=sum(F)
        F_array = (F_array-F_array.min())/(F_array.max()-F_array.min())
        F_rec[ii] = F_array
        res[ii] = ret

    return res,F_rec
