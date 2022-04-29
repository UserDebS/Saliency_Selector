# %% [markdown]
# # SLIC saliency
# 

# %%
import numpy as np
import pandas as pd
import cv2
import skimage
from skimage.transform import resize
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage.measure import regionprops
from skimage.filters import threshold_otsu


# %%
def computeSLICsaliency(img,N=512):

    # print("Getting SLIC superpixels...")
    segments = slic(img, n_segments=N, compactness=10,
            multichannel=True,
            enforce_connectivity=True,start_label=1)
    height,width,depth=img.shape

#     print("Computing Superpixels' Properties...")
    R_regions,G_regions,B_regions,R_xy,G_xy,B_xy,R_mi,G_mi,B_mi = getRegionInfo(segments,img)

    sal_R = np.zeros((height,width))
    sal_G = np.zeros((height,width))
    sal_B = np.zeros((height,width))
    
#     print("Computing Saliency Map For channel 1...")
    sal_R = mapChannelSaliency(R_regions,sal_R,R_xy,R_mi,height,width)
#     print("Computing Saliency Map For channel 2...")
    sal_G = mapChannelSaliency(G_regions,sal_G,G_xy,G_mi,height,width)
#     print("Computing Saliency Map For channel 3...")
    sal_B = mapChannelSaliency(B_regions,sal_B,B_xy,B_mi,height,width)
    
    SaliencyMap = sal_R+sal_G+sal_B
    SaliencyMap = ((SaliencyMap/SaliencyMap.max())*255).astype('int')
#     print("Computing Saliency Mask...")
    return SaliencyMap


# %%
def getRegionInfo(segments,img):
    R_regions = regionprops(segments, intensity_image=img[:,:,0])
    G_regions = regionprops(segments, intensity_image=img[:,:,1])
    B_regions = regionprops(segments, intensity_image=img[:,:,2])
    if (len(B_regions)==len(G_regions)==len(R_regions)):
        L = len(R_regions)
        # print(True)
    R_xy = np.zeros((L,2))
    G_xy = np.zeros((L,2))
    B_xy = np.zeros((L,2))
    R_mi = np.zeros(L)
    G_mi = np.zeros(L)
    B_mi = np.zeros(L)
    for i in range(len(R_regions)):
        R_xy[i] = R_regions[i].centroid
        G_xy[i] = G_regions[i].centroid
        B_xy[i] = B_regions[i].centroid
        R_mi[i] = int(R_regions[i].mean_intensity)
        G_mi[i] = int(G_regions[i].mean_intensity)
        B_mi[i] = int(B_regions[i].mean_intensity)
    return R_regions,G_regions,B_regions,R_xy,G_xy,B_xy,R_mi,G_mi,B_mi

# %%
def mapChannelSaliency(Regions,map,rg_xy,rg_mi,h,w):
    L = len(Regions)
    for i in range(L):
        denominator = np.sqrt(h**2+w**2)
        numerator = np.linalg.norm(rg_xy - rg_xy[i],axis=1)
        multiplier = np.linalg.norm(rg_mi-rg_mi[i])
        sal = np.sum(multiplier*(np.exp(-numerator/denominator)))
        coords_ = Regions[i].coords
        for coord in coords_:
            x = coord[0]
            y = coord[1]
            map[x][y] = sal
    return ((map/map.max())*255).astype('int')


