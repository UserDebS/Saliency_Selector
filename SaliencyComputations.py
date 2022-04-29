# %%
from kmeanssaliency1 import kMean_saliency
from slicSaliency import computeSLICsaliency
import cv2
import timeit
import joblib
import shutil
import os
from skimage.transform import resize

datasets = ["Cars","dog","sky","cultivated land",'cats']

# %%

def flushFolder(method,d):
    parent_dir = f"./Saliency/"
    directory = f"./{method}/saliency_{d}" 
    path = os.path.join(parent_dir, directory)
    try:
        shutil.rmtree(path)
    except OSError as e:
        print("Warning: %s - %s." % (e.filename, e.strerror)+"  rmtree not required!")
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)

# %%
def computeKMeansForDataset(d):
    method='KMeans'
    flushFolder(method,d)
    folder=f"./inputs/sample_{d}"
    files = os.listdir(folder)
    files.sort()
    i = 0
    for filename in files:
        img = cv2.imread(os.path.join(folder,filename))
        img = resize(img,(128,128,img.shape[2]))
        img =((img/img.max())*255).astype('uint8')
        SAlMap=kMean_saliency(img,85)
        cv2.imwrite(f'./Saliency/{method}/saliency_{d}/{i}.jpg',SAlMap)
        i+=1


# %%
def KmeansTimes():
  KMeans_times = {}
  n =1
  datasets = ["Cars","dog","sky","cultivated land",'cats']
  for d in datasets:
    t = timeit.timeit('computeKMeansForDataset(d)',globals=globals(),number=n)
    KMeans_times[d] = t/n
  joblib.dump(KMeans_times,'KMeans_times.pkl')

# %%


# %%
def fineGrained(d):
    method='FineGrained'
    flushFolder(method,d)
    folder=f"./inputs/sample_{d}"
    files = os.listdir(folder)
    files.sort()
    i = 0
    for filename in files:
      img = cv2.imread(os.path.join(folder,filename))
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      img = resize(img,(128,128,img.shape[2]))
      img =((img/img.max())*255).astype('uint8')
      # initialize OpenCV's static fine grained saliency detector and computing the saliency map
      saliency = cv2.saliency.StaticSaliencyFineGrained_create()
      (success, saliencyMap) = saliency.computeSaliency(img)
      # plt.imshow(saliencyMap,cmap='gray')
      # plt.show()
      cv2.imwrite(f'./Saliency/{method}/saliency_{d}/{i}.jpg',saliencyMap)
      i+=1
def fineGrainedTimes():
  FineGrained_times = {}
  n =10
  datasets = ["Cars","dog","sky","cultivated land",'cats']
  for d in datasets:
    t = timeit.timeit('fineGrained(d)',globals=globals(),number=n)
    FineGrained_times[d] = t/n
  joblib.dump(FineGrained_times,'times/FineGrained_times.pkl')

# %%


# %%
def spectralSaliency(d):
    method='Spectral'
    flushFolder(method,d)
    folder=f"./inputs/sample_{d}"
    files = os.listdir(folder)
    files.sort()
    i = 0
    for filename in files:
      img = cv2.imread(os.path.join(folder,filename))
      img = resize(img,(128,128,img.shape[2]))
      img =((img/img.max())*255).astype('uint8')
      # initialize OpenCV's static saliency spectral residual detector and computing the saliency map
      saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
      (success, saliencyMap) = saliency.computeSaliency(img)
      saliencyMap = (saliencyMap * 255).astype("uint8")
      cv2.imwrite(f'./Saliency/{method}/saliency_{d}/{i}.jpg',saliencyMap)
      i+=1

# %%
def SpectralTimes():
  Spectral_times = {}
  n =10
  datasets = ["Cars","dog","sky","cultivated land",'cats']
  for x in datasets:
    t = timeit.timeit('spectralSaliency(x)',globals=globals(),number=n)
    Spectral_times[d] = t/n
  joblib.dump(Spectral_times,'times/Spectral_times.pkl')

# %%


# %%
def computeSlicForDataset(d):
    method = 'SLIC'
    flushFolder(method,d)
    folder = f"./inputs/sample_{d}"
    files = os.listdir(folder)
    files.sort()
    i = 0
    for filename in files:
        img = cv2.imread(os.path.join(folder,filename))
        img = resize(img,(128,128,img.shape[2]))
        img =((img/img.max())*255).astype('uint8')
        SAlMap = computeSLICsaliency(img,2048)
        cv2.imwrite(f'./Saliency/{method}/saliency_{d}/{i}.jpg',SAlMap)
        i+=1
def SlicTimes():
    SLIC_times = {}
    n =10
    datasets = ["Cars","dog","sky","cultivated land",'cats']
    for x in datasets:
        t = timeit.timeit('computeSlicForDataset(x)',globals=globals(),number=n)
        SLIC_times[x] = t/n
    joblib.dump(SLIC_times,'times/SLIC_times.pkl')

# %%


# %%
def PoolNetTimes():
    PoolNet_times = {
        'Cars': 2.38,
        'dog':2.01,
        'sky':2.21,
        'cultivated land': 13.9,
        'cats':1.8
    }
    joblib.dump(PoolNet_times,'times/PoolNet_times.pkl')

# %%



if __name__ == '__main__':
  datasets = ["Cars","dog","sky","cultivated land",'cats']

  for d in datasets:
    computeSlicForDataset(d)
  SlicTimes()

  for d in datasets:
    spectralSaliency(d)
  SpectralTimes()

  for d in datasets:
    fineGrained(d)
  fineGrainedTimes()

  for d in datasets:
    computeKMeansForDataset(d)
  KmeansTimes()

  PoolNetTimes()
