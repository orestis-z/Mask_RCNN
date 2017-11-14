import sys, os
import numpy as np
from tqdm import tqdm

parent_dir = os.path.abspath("..")
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import tools


def compute_RGB_mean(path):
    file_list = os.listdir(path)
    n_files = len(file_list)
    R_mean = np.zeros(n_files)
    G_mean = np.zeros(n_files)
    B_mean = np.zeros(n_files)
    
    count = 0
    for i, file in enumerate(tqdm(file_list)):
        R, G, B, n = tools.compute_RGB_mean(os.path.join(path, file, 'image'))
        R_mean[i] = R * n
        G_mean[i] = G * n
        B_mean[i] = B * n
        count += n
    
    return np.array([np.sum(R_mean), np.sum(G_mean), np.sum(B_mean)]) / count

def compute_depth_mean(path):
    file_list = os.listdir(path)
    n_files = len(file_list)
    depth_mean = np.zeros(n_files)

    count = 0
    for i, file in enumerate(tqdm(file_list)):
        depth, n = tools.compute_depth_mean(os.path.join(path, file, 'depth'))
        depth_mean[i] = depth * n
        count += n
    
    return np.sum([depth_mean]) / count

if __name__ == "__main__":
    path = sys.argv[-1]
    rgb = compute_RGB_mean(path)
    print("\n RGB_mean:")
    print(rgb)
    d = compute_depth_mean(path)
    print("\n depth_mean:")
    print(d)
    print("\n RGBD_mean:")
    print(np.append(rgb, d))
