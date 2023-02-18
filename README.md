
This repo contains a binary classifier. \
This is the shortest wasy I could think of. \
The other option was to annotate the faces and the eyes use an object detect.
I scrapped from Google a small dataset of ~600 image of persons with open eyes and ! 300 images woth closed eyes.\
I chose the images the following way: single person, frontal gaze. A relatively close portrait.\
This is, I think, close your data. The dataset is by no meant comprehensive, somme images contain proprietary logs. I tried to filter them out as much I can,
but there are still some left. As you know, the human annotator error is about 5% :).\

The data is here: https://www.dropbox.com/sh/xesbel3lnbw8s8k/AAAfssBACmV__GHoH0ohHIpva?dl=0

 - faces_split.zip contains the actual inages
 - ffcv_dataset.zip contains the preprocessed data for training. Please unzip and copy both .beton file to the data folder in the repo.

# What has been done:
  * dataset scrapping a filtering
  * train and inference scripts - sue weighted F1 score and confusion matrix for evaluation. \
    The F1 favours recall since I though ti is better to get all the blurred images at the expense of some oversensitivity.\
  

# Ho to run the repo:
   *  download and unzip ffcv_dataset.zip and copy both files to data folder.

# Features:
 - **pytorch-lightning**  - for rapid prototyping. \
   It saves a lot of time and a lot of boilerplate code. \
   It also has tons of options for senility check, learning rate finder, automatic batch size optimization, dataset partitioning, precision and many others.\
 - **ffcv** - it is a pain to install, but it can speed up a pytoch dataloader x10 at least.
 - **timm** -  for easy backbones switching. It contains the largest implemented set of net with a common API. Priceless.
 - **torchmetrics** - thread-safe speed optimized metrics.
 - **progressive image size** - for faster training. It also makes the model more robust since it sees different image sizes
 - **centralized config** - they are all summarized in config.yaml. 
 - Sometimes the parameters are so many, it is tedious to look for one when they are all over the project.
 - **experiments logging** - the workdir folder contains all the data from the current run: best checkpoint, the console log, train.py and config.yaml
 - **Weights $ Biases logging** - much better from tensorboard. For models performance comparison. It is disabled since you need a wandb account.


# How I would have solved both tasks:
Use a single object detector for the faces and the eyes. The eyes would be two classes: open and closed.
Use the face crop to estimate the blur.
Or detect the whole body instead of the face and use it for blur detection.