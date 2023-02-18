import random
import kornia as K
import torch
import timm
import numpy as np
import torchvision.transforms.functional as F
from omegaconf import OmegaConf
from wcmatch.pathlib import Path
from torchvision.io import read_image
from matplotlib import pyplot as plt


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].axis('off')
        axs[0, i].set_title(f'{labels[i]}', fontsize=10)
    plt.show()



if __name__ == '__main__':
    IMAGE_FOLDER = '/data1/riverside/faces_split/val'
    CHECKPOINT_FILE = 'workdir/train/exp_16-Feb-2023_00_39_12/epoch_epoch=43_vfb_v_fb=1.0000.ckpt'
    LABELS_DICT = {0: 'open_eyes', 1: 'closed_eyes'}

    image_path = Path(IMAGE_FOLDER)
    conf = OmegaConf.load('config.yaml')
    model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=conf.train.num_classes)
    checkpoint = torch.load(CHECKPOINT_FILE)
    transforms = torch.nn.Sequential(
        K.augmentation.Resize(conf.img.size),
        K.augmentation.Normalize(mean=conf.img.mean,
                                 std=conf.img.std)
    )

    state_dict_stripped = dict()
    for x in checkpoint['state_dict']:
        state_dict_stripped[x.replace('model.', '')] = checkpoint['state_dict'][x]


    model.load_state_dict(state_dict_stripped)
    model.cuda()
    model.eval()

    images = list(image_path.rglob(['*.png', '*.jpeg', '*.jpg']))
    rand_idxs =random.choices(range(len(images)), k=5)
    images = [images[i] for i in rand_idxs]
    labels = []
    im_tensors = []
    for image in images:
        with torch.no_grad():
            im_tensor = read_image(image.as_posix())
            im_tensors.append(im_tensor)
            im_tensor = im_tensor.unsqueeze(0).float().cuda()

            preds = model(im_tensor)
            label = LABELS_DICT[torch.argmax(preds).item()]
            labels.append(label)

    show(im_tensors)