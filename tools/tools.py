import os, sys
import zipfile
from skimage import io
import numpy as np
from pprint import pprint
from tqdm import tqdm

def unzip(file, dest):
    zip_ref = zipfile.ZipFile(file, 'r')
    zip_ref.extractall(dest)
    zip_ref.close()

def shrink(path, freq):
    for root, dirs, files in os.walk(path):
        print('Found directory: %s' % root)
        for i, file in enumerate(sorted(files)):
            if i % freq != 0 and file[-4:] == '.png':
                os.remove(os.path.join(root, file))

def compute_RGB_mean(path):
    file_list = os.listdir(path)
    n_files = len(file_list)
    R_mean = np.zeros(n_files)
    G_mean = np.zeros(n_files)
    B_mean = np.zeros(n_files)
    
    count = 0
    for i, file in enumerate(tqdm(file_list)):
        assert(file[-4:] in ['.jpg', '.png'])
        file_path = os.path.join(path, file)
        if (os.stat(file_path).st_size):
            img = io.imread(file_path)
            R_mean[i] = np.mean(img[:, :, 0])
            G_mean[i] = np.mean(img[:, :, 1])
            B_mean[i] = np.mean(img[:, :, 2])
            count += 1
        else:
            R_mean[i] = np.nan
            G_mean[i] = np.nan
            B_mean[i] = np.nan

    return (np.nanmean(R_mean), np.nanmean(G_mean), np.nanmean(B_mean), count)

def compute_depth_mean(path, depth=False):
    file_list = os.listdir(path)
    n_files = len(file_list)
    D_mean = np.zeros(n_files)
    
    count = 0
    for i, file in enumerate(tqdm(file_list)):
        assert(file[-4:] in ['.jpg', '.png'])
        file_path = os.path.join(path, file)
        if (os.stat(file_path).st_size):
            img = io.imread(file_path)
            D_mean[i] = np.mean(img)
            count += 1
        else:
            D_mean[i] = np.nan

    return (np.nanmean(D_mean), count)

def count_files(path):
    n_files = []
    sub_directories = []
    for root, dirs, files in os.walk(path):
        sub_directories.append(root)
        n_files.append(len(files))
    pprint(sorted(list(zip(sub_directories, n_files))))
