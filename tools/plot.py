import json
import matplotlib.pyplot as plt
# from scipy.signal import savgol_filter
from scipy.ndimage.filters import gaussian_filter1d
import numpy as np


with open('data/run_scenenet_coco_rgb20180428T1942-tag-loss.json') as f:
    data = json.load(f)
    steps = [el[1] for el in data]
    loss = np.array([el[2] for el in data])
data_rgb_train = [steps, loss]

with open('data/run_scenenet20180428T1942-tag-loss.json') as f:
    data = json.load(f)
    steps = [el[1] for el in data]
    loss = np.array([el[2] for el in data])
data_rgbd_train = [steps, loss]

with open('data/run_scenenet_coco_rgb20180428T1942-tag-val_loss.json') as f:
    data = json.load(f)
    steps = [el[1] for el in data]
    loss = np.array([el[2] for el in data])
data_rgb_val = [steps, loss]

with open('data/run_scenenet20180428T1942-tag-val_loss.json') as f:
    data = json.load(f)
    steps = [el[1] for el in data]
    loss = np.array([el[2] for el in data])
data_rgbd_val = [steps, loss]

# smooth = lambda data: data
smooth = lambda data, sigma=3: gaussian_filter1d(data, sigma)
# smooth = lambda data: savgol_filter(data, 51, 5)

color_1 = (31 / 255, 119 / 255, 180 / 255)
color_2 = (1, 127 / 255, 14 / 255)
lw = 1
ylim = (0.8, 1.8)

fig = plt.figure()

ax_1 = fig.add_subplot(121)
ax_1.plot(data_rgb_train[0], data_rgb_train[1], color=color_1 + (0.5,), lw=lw)
p1 = ax_1.plot(data_rgb_train[0], smooth(data_rgb_train[1]), color=color_1, lw=lw, label="RGB")
ax_1.plot(data_rgbd_train[0], data_rgbd_train[1], color=color_2 + (0.5,), lw=lw)
p2 = ax_1.plot(data_rgbd_train[0], smooth(data_rgbd_train[1]), color=color_2, lw=lw, label="RGB-D")
ax_1.set_title("Training")
ax_1.legend()
# ax_1.ylabel('loss')
ax_1.grid()
ax_1.set_ylim(ylim)

ax_2 = fig.add_subplot(122)
sigma = 5
ax_2.plot(data_rgb_val[0], data_rgb_val[1], color=color_1 + (0.5,), lw=lw)
p1 = ax_2.plot(data_rgb_val[0], smooth(data_rgb_val[1], sigma), color=color_1, lw=lw, label="RGB")
ax_2.plot(data_rgbd_val[0], data_rgbd_val[1], color=color_2 + (0.5,), lw=lw)
p2 = ax_2.plot(data_rgbd_val[0], smooth(data_rgbd_val[1], sigma), color=color_2, lw=lw, label="RGB-D")
ax_2.set_title("Validation")
# ax_2.legend()
# ax_2.ylabel('loss')
ax_2.grid()
ax_2.set_ylim(ylim)
# ax_2.axes.get_yaxis().set_ticklabels([])

fig.text(0.5, 0.04, 'Epoch', ha='center', va='center')
fig.text(0.06, 0.5, 'Loss', ha='center', va='center', rotation='vertical')

plt.show()

fig.savefig('loss.png', dpi=300, bbox_inches='tight')

