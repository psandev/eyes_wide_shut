1. You need to create a conda virtual environment. It has to be conda due to two libraries, which cannot be installed in a pip environment: ffcv and xgboost-gpu.

    If you do not have conda installed, I suggest you use miniconda, the conda slimmed-down version:
    Go to the page below and follow instructions. It is a two-step process: 
    download and install the bash script.\
    <br>
    https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html
    <br>
    <br>
2. **The libraries installations  for BOTH projects**:
    <br>
    <br>
   * **conda create -n riverside_39 python=3.9 cupy pkg-config compilers libjpeg-turbo opencv pytorch torchvision pytorch-cuda=11.7 numba -c conda-forge -c nvidia -c pytorch && conda activate riverside_39 && conda update ffmpeg && pip install ffcv** \
    <br>
   This line creates an env and install ffcv and other libraries. This can take 10-20-30 min depending on your connection. Sorry.\
   I hope you have a current cuda toolkit installed: 11.5, 11.6 or 11.7. If the installation fails, talk to me.\
   <br>
   * **conda activate riverside_39**\
   <br>
    PLEASE CHECK YOUR TORCH INSTALLATION IF IT IS CUDA COMPATIBLE.\
    In a terminal:\
    **python\
    import torch\
    torch.cuda.is_available()** - if the answer is True, you are good to go.
    <br>
   <br>
   * **conda install -c conda-forge py-xgboost-gpu**
   <br>
   <br>
   * **pip install wcmatch matplotlib scikit-learn scipy opencv-python blurgenerator scikit-image kornia torch-dct optuna torchmetrics==0.7.3 pytorch-lightning loguru omegaconf timm tqdm wandb**
   <br>
   <br>
   * And you are done. I decided on this uproach because pip install -r requirement.txt kept crashing.
    