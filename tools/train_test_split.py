import splitfolders


if __name__ == '__main__':
    FOLDER_IN = '/media/peter/NVME980/riverside/faces'
    FOLDER_OUT = '/media/peter/NVME980/riverside/faces_split'

    splitfolders.ratio(FOLDER_IN, output=FOLDER_OUT,
                       seed=42, ratio=(.9, .1),
                       group_prefix=None,
                       move=False)