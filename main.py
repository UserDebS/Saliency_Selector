# %% [markdown]
# # Discounted Saliency Selector

# %%
from skimage.transform import resize
from skimage.filters import threshold_otsu
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from PIL import Image
import itertools
import os
from skimage.color import rgb2gray
from sklearn.preprocessing import StandardScaler
from math import erf, sqrt
import scipy
def load_images_from_folder(folder,RGB=True,size = 128):
    images = []
    files = os.listdir(folder)
    files.sort()
    for filename in files:
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            if RGB:
                img = resize(img,(size,size,3))
            else:
                img = rgb2gray(img)
                img = resize(img,(size,size))
            img =((img/img.max())*255).astype('uint8')
            images.append(img)
    return np.array(images)

# %%
datasets = ["Cars","dog","sky","cultivated land"]
methods = ['SLIC','Spectral','PoolNet','KMeans','FineGrained']

# %%
ip = input(f"""Which dataset you want to chose:
1. {datasets[0]}
2. {datasets[1]}
3. {datasets[2]}
4. {datasets[3]}\n""")

if(int(ip) > len(datasets)): 
    print('Please enter a valid input')
    exit(0)

# %%
d = datasets[int(ip)-1]

# %%
print(f'Dataset Selected: {d}')

# %%
def GetSalMaps(d):
    methods = ['SLIC','Spectral','PoolNet','KMeans','FineGrained']
    SaliencyMaps = {}
    for method in methods:
        folder = f"./Saliency/{method}/saliency_{d}"
        files = os.listdir(folder)
        files.sort()
        SaliencyMaps[method] = load_images_from_folder(folder,RGB=False)
    return SaliencyMaps
SaliencyMaps = GetSalMaps(d)


# %%
def getDistro(bins,hist):
    D = []
    for i in range(len(hist)):
        D.extend([bins[i]]*hist[i])
    return np.array(D)
def normalise(Distr):
    return (Distr - Distr.min())/(Distr.max()-Distr.min())
def normal_dist(x , mean , sd):
    prob_density = (np.pi*sd) * np.exp(-0.5*((x-mean)/sd)**2)
    return prob_density   

# %% [markdown]
# http://modelai.gettysburg.edu/2020/wgan/Resources/Lesson1/kl-divergence-gaussians.htm

# %% [markdown]
# ## Separation Analysis 
# Separation analysis between the foreground distribution and background distribution of the saliency map  
# using **Kullback–Leibler divergence** between the background pixel distribution and foreground distribution

# %%
def kld(masks):
    # """returns mean and standard deviation of background and foreground"""  
    KLD = []
    for m in masks:

        t = (threshold_otsu(m))
        hist,bins = np.histogram(m.ravel(),256)
        
        N_bg, N_fg = hist[:t].sum(),hist[t:].sum()

        pixels = hist*bins[:256]

        D_bg = normalise(getDistro(bins[:t],hist[:t]))
        D_fg = normalise(getDistro(bins[t:256],hist[t:]))

                
        bg_mean = D_bg.mean()
        fg_mean = D_fg.mean()
                
        bg_sd = np.std(D_bg)
        fg_sd = np.std(D_fg)

        t1 = np.log10(bg_sd/fg_sd)
        t2 = (fg_sd**2 + (fg_mean-bg_mean)**2)/(2*bg_sd**2)
        KLD.append(abs(t1 + t2 - 0.5))


    return np.array(KLD)

# %%
KLDs = {}
for i in SaliencyMaps:
    KLDs[i] = kld(SaliencyMaps[i])

# %% [markdown]
# ## connected component analysis - Concentration measure
# opencv's connectedComponentsWithStats() function is being used with 4 as connected component  
# parameter to obtain the salient foreground objects and their area.  
# This area of each blob is then used to compute the concentration measure of the Saliency maps  
# 

# %%
def getConcentration(maps):
    C = []
    for d in maps:
        binary = (d>threshold_otsu(d)).astype('uint8')
        (numLabels, labels, stats, centroids) = cv2.connectedComponentsWithStats(binary,4,cv2.CV_32S)
        areas = np.zeros(numLabels)
        for i in range(0, numLabels):
            areas[i]=stats[i, cv2.CC_STAT_AREA]
        Cu = areas/np.sum(areas)
        u = np.argmax(Cu)
        concentration = Cu[u] + ((1-Cu[u])/numLabels)
        C.append(concentration)
    return np.array(C)

# %% [markdown]
# Concetration measure and Separation measure are multiplied to get the final saliency score for each saliency map.  

# %%
CMs = {}
for i in SaliencyMaps:
    CMs[i] = getConcentration(SaliencyMaps[i])
CMs

# %%
Quality_Score = {}
for method in methods:
    Quality_Score[method] = np.sum(np.log10(KLDs[method])+CMs[method])/5

# %%
Quality_Score

# %%
times = {}
for method in methods:
    times[method] = joblib.load(f"times/{method}_times.pkl")
time = {method:times[method][d] for method in methods}
time

# %%
T_max = max(time.values())
T_min = min(time.values())
Q_max = max(Quality_Score.values())

# %%
def timemesure(time):
    return np.log10((T_max-time+0.5)/(time-T_min+0.5))

# %%
scores = {}
for method in methods:
    Q = Quality_Score[method]
    scores[method] = Q + (Q/Q_max)*timemesure(time[method])


# %%
Sorted_Scores = { i[0]:i[1] for i in sorted(scores.items(),key=lambda x:x[1],reverse=True)}
Sorted_Scores

# %%
print(f"The best method is: {list(Sorted_Scores.keys())[0]}")

# %%
print('Methods in order of their preferences:\n')
for i in Sorted_Scores:
    print(f'{i} with score {Sorted_Scores[i]}')


