import cv2
from wcmatch.pathlib import Path
from torch.utils.data import Dataset
from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, IntField



class MyDataset(Dataset):
    def __init__(self, img_folder, img_size, stage):
        assert stage in ['train', 'val', 'test'], f'type could be train, val, or test not {type}'

        path_in = Path(img_folder)
        self._img_size = img_size
        self._img_list = list((path_in/stage).rglob(['*.jpg', '*.jpeg', '*.png']))
        self._label_map = {'open_eyes': 0, 'closed_eyes': 1}


    def __getitem__(self, index):
        image_path = self._img_list[index]
        image = cv2.imread(image_path.as_posix())
        if not image.shape[:-1] == self._img_size:
            image = cv2.resize(image, self._img_size, interpolation=cv2.INTER_LANCZOS4)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = image_path.parts[-2]
        label = self._label_map[label]
        return image, label

    def __len__(self):
        return len(self._img_list)


if __name__ == '__main__':
    IMAGE_FOLDER = '/data1/riverside/faces_split'
    FOLDER_OUT = '/data1/riverside/ffcv_dataset'
    IMG_SIZE = (640, 640)

    path_image_folder = Path(IMAGE_FOLDER)
    path_out = Path(FOLDER_OUT)
    if not path_out.exists():
        path_out.mkdir()

    train_ds = MyDataset(path_image_folder, IMG_SIZE, 'train')
    val_ds = MyDataset(path_image_folder, IMG_SIZE, 'val')
    datasets = {'train': train_ds, 'val': val_ds}

    for (name, ds) in datasets.items():
        writer = DatasetWriter(path_out/f'{name}_ds.beton', {
            'image': RGBImageField(write_mode='raw'),
            # 'image': NDArrayField(shape=(3,32,32), dtype=np.dtype('<u1')),
            'label': IntField()
        })
        writer.from_indexed_dataset(ds)