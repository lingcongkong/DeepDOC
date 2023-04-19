import os
import pandas as pd
import numpy as np
import shutil
import nibabel as nib


def resize3d(img, shape):
    x, y, z = shape
    t_img = []
    f_img = []
    for i in img:
        t_img.append(cv.resize(i, (z, y)))
    t_img = np.float32(t_img)
    for i in range(t_img.shape[1]):
        f_img.append(cv.resize(t_img[:, i], (x, z)))
    f_img = np.float32(f_img)
    f_img = np.transpose(f_img, (1, 0, 2))
    return f_img



def cvt(img):
    fg = cv.applyColorMap(np.uint8(255 * img), cv.COLORMAP_JET)
    return fg


def get(ana, img, ax, idx):
    if ax == 'x':
        ana = ana[idx]
        img = img[idx]
    elif ax == 'y':
        ana = ana[:, idx]
        img = img[:, idx]
    else:
        ana = ana[:, :, idx]
        img = img[:, :, idx]
    m = ana.max()
    bg = np.repeat(ana, 3).reshape(ana.shape[0], ana.shape[1], -1)/m
    fg = cvt(img)
    return bg, fg


def visualize_gradcam(data_path, gradcam_path, pid):
    ana = np.array(nib.load(f'{data_path}/preprocessed/p{pid}_anat.nii').dataobj)
    ana[np.isnan(ana)] = 0
    ana = ana/ana.max()
    i = '0'+str(i) if len(str(i)) == 1 else str(i)
    img = resize3d(np.array(nib.load('{}/p{}_gc.nii'.format(gradcam_path, i)).dataobj), (181, 217, 181))
    img = img/img.max()
    os.makedirs('{}/p{}'.format(gradcam_path, i), exist_ok=True)
#     print(bg.max())
    for x in range(181):
        bg, fg = get(ana, img, 'x', x)
        cv.imwrite('{}/p{}/x{}.png'.format(gradcam_path, i, x), cv.rotate(bg*0.7+fg/255*0.3, 2)*255)