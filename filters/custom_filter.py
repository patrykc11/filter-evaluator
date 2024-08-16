# https://learnopencv.com/improving-illumination-in-night-time-images/

import os
import shutil
import cv2
import numpy as np
from PIL import Image
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Implementation for Guided Image Filtering
Reference:
http://research.microsoft.com/en-us/um/people/kahe/eccv10/
"""
from itertools import combinations_with_replacement
from collections import defaultdict
import time
from numpy.linalg import inv

R, G, B = 0, 1, 2  # index for convenience

# 1. Getting dark or light illumination channel of image
def get_illumination_channel(I, w):
    M, N, _ = I.shape
    # padding for channels
    padded = np.pad(I, ((int(w/2), int(w/2)), (int(w/2), int(w/2)), (0, 0)), 'edge')
    darkch = np.zeros((M, N))
    brightch = np.zeros((M, N))
 
    for i, j in np.ndindex(darkch.shape):
        darkch[i, j] = np.min(padded[i:i + w, j:j + w, :]) # dark channel
        brightch[i, j] = np.max(padded[i:i + w, j:j + w, :]) # bright channel
 
    return darkch, brightch

# 2. Computing Global Atmosphere Lighting
def get_atmosphere(I, brightch, p=0.3, threshold=100):
    M, N = brightch.shape
    flatI = I.reshape(M*N, 3)  # reshaping image array
    flatbright = brightch.ravel()  # flattening image array

    searchidx = (-flatbright).argsort()[:int(M*N*p)] # sorting and slicing
    A = np.mean(flatI.take(searchidx, axis=0), dtype=np.float64, axis=0)
    return A

# 3. Finding the Initial Transmission Map
def get_initial_transmission(A, brightch):
    A_c = np.max(A)
    init_t = (brightch-A_c)/(1.-A_c) # finding initial transmission map
    return (init_t - np.min(init_t))/(np.max(init_t) - np.min(init_t)) # normalized initial transmission map

# 4. Using Dark Channel to Estimate Corrected Transmission Map
def get_corrected_transmission(I, A, darkch, brightch, init_t, alpha, omega, w):
    im = np.empty(I.shape, I.dtype);
    for ind in range(0, 3):
        im[:, :, ind] = I[:, :, ind] / A[ind] #divide pixel values by atmospheric light
    dark_c, _ = get_illumination_channel(im, w) # dark channel transmission map
    dark_t = 1 - omega*dark_c # corrected dark transmission map
    corrected_t = init_t # initializing corrected transmission map with initial transmission map
    diffch = brightch - darkch # difference between transmission maps
 
    for i in range(diffch.shape[0]):
        for j in range(diffch.shape[1]):
            if(diffch[i, j] < alpha):
                corrected_t[i, j] = dark_t[i, j] * init_t[i, j]
 
    return np.abs(corrected_t)

def boxfilter(I, r):
    """Fast box filter implementation.

    Parameters
    ----------
    I:  a single channel/gray image data normalized to [0.0, 1.0]
    r:  window radius

    Return
    -----------
    The filtered image data.
    """
    M, N = I.shape
    dest = np.zeros((M, N))
    #print(I)
    
    # cumulative sum over Y axis (tate-houkou no wa)
    sumY = np.cumsum(I, axis=0)
    #print('sumY:{}'.format(sumY))
    # difference over Y axis
    dest[:r + 1] = sumY[r:2*r + 1] # top r+1 lines
    dest[r + 1:M - r] = sumY[2*r + 1:] - sumY[:M - 2*r - 1]
    #print(sumY[2*r + 1:]) # from 2*r+1 to end lines
    #print(sumY[:M - 2*r - 1]) # same lines of above, from start
    #tile replicate sumY[-1] and line them up to match the shape of (r, 1)
    dest[-r:] = np.tile(sumY[-1], (r, 1)) - sumY[M - 2*r - 1:M - r - 1] # bottom r lines

    # cumulative sum over X axis
    sumX = np.cumsum(dest, axis=1)
    #print('sumX:{}'.format(sumX))
    # difference over X axis
    dest[:, :r + 1] = sumX[:, r:2*r + 1] # left r+1 columns
    dest[:, r + 1:N - r] = sumX[:, 2*r + 1:] - sumX[:, :N - 2*r - 1]
    dest[:, -r:] = np.tile(sumX[:, -1][:, None], (1, r)) - sumX[:, N - 2*r - 1:N - r - 1] # right r columns

    #print(dest)

    return dest

# 5. Smoothing Transmission Map using Guided Filter
def guided_filter(I, p, r=15, eps=1e-3):
    """Refine a filter under the guidance of another (RGB) image.

    Parameters
    -----------
    I:   an M * N * 3 RGB image for guidance.
    p:   the M * N filter to be guided. transmission is used for this case.
    r:   the radius of the guidance
    eps: epsilon for the guided filter

    Return
    -----------
    The guided filter.
    """
    M, N = p.shape
    base = boxfilter(np.ones((M, N)), r) # this is needed for regularization
    
    # each channel of I filtered with the mean filter. this is myu.
    means = [boxfilter(I[:, :, i], r) / base for i in range(3)]
    
    # p filtered with the mean filter
    mean_p = boxfilter(p, r) / base

    # filter I with p then filter it with the mean filter
    means_IP = [boxfilter(I[:, :, i]*p, r) / base for i in range(3)]

    # covariance of (I, p) in each local patch
    covIP = [means_IP[i] - means[i]*mean_p for i in range(3)]

    # variance of I in each local patch: the matrix Sigma in ECCV10 eq.14
    var = defaultdict(dict)
    for i, j in combinations_with_replacement(range(3), 2):
        var[i][j] = boxfilter(I[:, :, i]*I[:, :, j], r) / base - means[i]*means[j]

    a = np.zeros((M, N, 3))
    for y, x in np.ndindex(M, N):
        #         rr, rg, rb
        # Sigma = rg, gg, gb
        #         rb, gb, bb
        Sigma = np.array([[var[R][R][y, x], var[R][G][y, x], var[R][B][y, x]],
                          [var[R][G][y, x], var[G][G][y, x], var[G][B][y, x]],
                          [var[R][B][y, x], var[G][B][y, x], var[B][B][y, x]]])
        cov = np.array([c[y, x] for c in covIP])
        a[y, x] = np.dot(cov, inv(Sigma + eps*np.eye(3)))  # eq 14

    # ECCV10 eq.15
    b = mean_p - a[:, :, R]*means[R] - a[:, :, G]*means[G] - a[:, :, B]*means[B]

    # ECCV10 eq.16
    q = (boxfilter(a[:, :, R], r)*I[:, :, R] + boxfilter(a[:, :, G], r)*I[:, :, G] + boxfilter(a[:, :, B], r)*I[:, :, B] + boxfilter(b, r)) / base

    return q

# 6. Calculating the Resultant Image
def get_final_image(I, A, refined_t, tmin):
    refined_t_broadcasted = np.broadcast_to(refined_t[:, :, None], (refined_t.shape[0], refined_t.shape[1], 3)) # duplicating the channel of 2D refined map to 3 channels
    J = (I-A) / (np.where(refined_t_broadcasted < tmin, tmin, refined_t_broadcasted)) + A # finding result 
 
    return (J - np.min(J))/(np.max(J) - np.min(J)) # normalized im

# reduce these intense spots of white
def reduce_init_t(init_t):
    init_t = (init_t*255).astype(np.uint8) 
    xp = [0, 32, 255]
    fp = [0, 32, 48]
    x = np.arange(256) # creating array [0,...,255]
    table = np.interp(x, xp, fp).astype('uint8') # interpreting fp according to xp in range of x
    init_t = cv2.LUT(init_t, table) # lookup table
    init_t = init_t.astype(np.float64)/255 # normalizing the transmission map
    return init_t

# final step
def dehaze(im, tmin=0.1, w=15, alpha=0.4, omega=0.75, p=0.1, eps=1e-3, reduce=False):
    I = np.asarray(im, dtype=np.float64) # Convert the input to a float array.
    I = I[:, :, :3] / 255
    m, n, _ = I.shape
    Idark, Ibright = get_illumination_channel(I, w)
    A = get_atmosphere(I, Ibright, p)
 
    init_t = get_initial_transmission(A, Ibright) 
    if reduce:
        init_t = reduce_init_t(init_t)
    corrected_t = get_corrected_transmission(I, A, Idark, Ibright, init_t, alpha, omega, w)
 
    normI = (I - I.min()) / (I.max() - I.min())
    refined_t = guided_filter(normI, corrected_t, w, eps) # applying guided filter
    J_refined = get_final_image(I, A, refined_t, tmin)
     
    enhanced = (J_refined*255).astype(np.uint8)
    f_enhanced = cv2.detailEnhance(enhanced, sigma_s=10, sigma_r=0.15)
    f_enhanced = cv2.edgePreservingFilter(f_enhanced, flags=1, sigma_s=64, sigma_r=0.2)
    return f_enhanced

input_folder = 'images/original_night'
output_folder = 'images/custom_filter_images'

if os.path.exists(output_folder):
    shutil.rmtree(output_folder)

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith(('.jpg', '.png', '.jpeg')):
        image_path = os.path.join(input_folder, filename)
        image = Image.open(image_path).convert('RGB')
        image_array = np.array(image)
        start_time = time.time()
        image_after_filter = dehaze(image_array, reduce=True)
        conversion_time = time.time() - start_time
        image_after_filter = cv2.cvtColor(image_after_filter, cv2.COLOR_RGB2BGR)
        output_image_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_image_path, image_after_filter)
        print(f"Conversion time for {filename}: {conversion_time:.3f} seconds")