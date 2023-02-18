"""
calcculate mean and std of all images

"""


import cv2
import numpy as np
from wcmatch.pathlib import Path
from tqdm import tqdm



if __name__ == '__main__':
    FOLDER = '/data1/riverside/faces_split'
    path = Path(FOLDER)
    images = list(path.rglob(['*.jpg', '*.jpeg', '*.png']))

    mean_list, std_list = [], []
    for img in tqdm(images, total=len(images)):
        img_arr = cv2.cvtColor(cv2.imread(img.as_posix()), cv2.COLOR_BGR2RGB)
        # img_arr = img_arr[..., 0] / 4 + img_arr[..., 1] / 2 + img_arr[..., 2] / 4
        mean_list.append(img_arr.mean(axis=(0, 1)).tolist())
        std_list.append(img_arr.std(axis=(0, 1)).tolist())

    mean_dataset = (np.vstack(mean_list).mean(axis=0)).round(3)
    std_dataset = (np.vstack(std_list).mean(axis=0)).round(3)
    print(f'mean: {mean_dataset}, std: {std_dataset}')
