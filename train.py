import torch
import wandb
import shutil
import timm
import kornia as K
import numpy as np
from pathlib import Path
import torchmetrics
import pytorch_lightning as pl

from loguru import logger
from omegaconf import OmegaConf
from datetime import datetime
from torch.nn import functional as F
from torch import nn, optim
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, Squeeze, NormalizeImage, \
    RandomHorizontalFlip, ToTorchImage, ImageMixup
from ffcv.fields.rgb_image import  RandomResizedCropRGBImageDecoder
from ffcv.fields.basics import IntDecoder

torch.set_printoptions(sci_mode=False, linewidth=200)



class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2,alpha=0.25, reduction='mean'):
        super(FocalLoss, self).__init__(weight,reduction=reduction)
        self.gamma = gamma
        self.alpha = alpha
        self.weight = weight #weight parameter will act as the alpha parameter to balance class weights

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target,reduction=self.reduction,weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = (self.alpha * (1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss



class DataModule(pl.LightningDataModule):
    def __init__(self, train_dl, val_dl,  conf: OmegaConf):
        super().__init__()
        self.train_dl = train_dl
        self.val_dl = val_dl

    def train_dataloader(self):
        return self.train_dl

    def val_dataloader(self):
        return  self.val_dl



class PL_NET(pl.LightningModule):
    def __init__(self, model, optimizer, criterion, conf, train_dl, val_dl):
        super().__init__()
        self.conf = conf
        self.model = model
        self.optim = optimizer
        self.criterion = criterion
        # self.transforms = transforms
        self.train_acc = torchmetrics.Accuracy(num_classes=conf.train.num_classes).to(ffcv_device)
        self.val_acc = torchmetrics.Accuracy(num_classes=conf.train.num_classes).to(ffcv_device)
        self.train_fbeta = torchmetrics.FBeta(num_classes=conf.train.num_classes, average='macro', beta=0.5)
        self.val_fbeta = torchmetrics.FBeta(num_classes=conf.train.num_classes, average='macro', beta=0.5)
        self.train_conf = torchmetrics.ConfusionMatrix(num_classes=conf.train.num_classes, dist_sync_on_step=True,
                                                       normalize='true')
        self.val_conf = torchmetrics.ConfusionMatrix(num_classes=conf.train.num_classes, dist_sync_on_step=True,
                                                     normalize='true')
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.validation_step_preds = []
        self.validation_step_labels = []
        self.training_step_preds = []
        self.training_step_labels = []




    def forward(self, x):
        return self.model(x)


    def train_dataloader(self):
        return self.train_dl


    def val_dataloader(self):
        return  self.val_dl

    def on_train_epoch_start(self):
        res = self.get_resolution(self.current_epoch)
        self.crop = (conf.img.size[0] - res)//2
        self.transforms = torch.nn.Sequential(
            # K.augmentation.ColorJitter(0.1, 0.1, 0.1, 0.1),
            # # K.color.RgbToGrayscale(),
            # K.filters.GaussianBlur2d((5, 5), (1, 1), border_type='constant'),
            K.augmentation.CenterCrop((res, res), p=1)
        )


    def training_step(self, batch, batch_idx):
        labels, data = batch[1], batch[0]
        if self.crop > 0:
            data = data[..., self.crop:-self.crop, self.crop:-self.crop]
        # data = self.transforms(data)
        self.res = data.shape[2]
        preds = self.model(data)
        loss = self.criterion(preds, labels)
        self.training_step_preds.append(preds)
        self.training_step_labels.append(labels)
        self.train_fbeta.update(preds.argmax(dim=1), labels)
        self.log('t_fb', self.train_fbeta, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('t_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss, 'preds': preds.detach(), 'labels': labels.detach()}



    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        items.pop("loss", None)
        return items


    def validation_step(self, batch, batch_idx):
        labels, data = batch[1], batch[0]
        if self.crop>0:
            data = data[..., self.crop:-self.crop, self.crop:-self.crop]
        # data = self.transforms(data)
        preds = self.model(data)
        self.validation_step_preds.append(preds)
        self.validation_step_labels.append(labels)
        loss = self.criterion(preds, labels)
        self.val_fbeta(preds.argmax(dim=1), labels)
        self.log('v_fb', self.val_fbeta, on_step=True, prog_bar=True, on_epoch=True, logger=True)
        self.log('v_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss, 'preds': preds.detach(), 'labels': labels.detach()}


    def on_train_epoch_end(self) -> None:
        if self.current_epoch % conf.val.every_n_epoch == 0 and self.current_epoch > 0:
            preds = torch.cat([x for x in self.training_step_preds])
            labels = torch.cat([x for x in self.training_step_labels])
            conf_mat = self.train_conf(preds, labels)
            logger.opt(colors=True).info(f'\nEpoch={self.current_epoch}, Train Confusion matrix:\n {conf_mat}')
        self.training_step_preds.clear()
        self.training_step_labels.clear()


    def on_validation_epoch_end(self) -> None:
        if self.current_epoch % conf.val.every_n_epoch == 0 and self.current_epoch > 0:
            val_fbeta = self.trainer.logged_metrics['v_fb_epoch']
            train_fbeta = self.trainer.logged_metrics['t_fb_epoch']
            val_loss = self.trainer.logged_metrics['v_loss_epoch']
            train_loss = self.trainer.logged_metrics['t_loss_epoch']
            lr = self.scheduler.get_last_lr()[0]
            logger.opt(colors=True).info(f'Epoch={self.current_epoch:03}/{conf.train.epochs}, '
                                         f'im_size={self.res}, '
                                         f'val_fbeta={val_fbeta:.4f}, train_fbeta={train_fbeta:.4f}, '
                                         f'val_loss={val_loss:.7f}, train_loss={train_loss:.7f}, '
                                         f'lr_rate={lr:.5f}')

            preds = torch.cat([x for x in self.validation_step_preds])
            labels = torch.cat([x for x in self.validation_step_labels])
            conf_mat = self.train_conf(preds, labels)
            logger.opt(colors=True).info(f'\nEpoch={self.current_epoch}, Val Confusion matrix:\n {conf_mat}')
        self.validation_step_preds.clear()
        self.validation_step_labels.clear()


    def configure_optimizers(self):
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optim, milestones=self.conf.optim.milestones,
                                                               gamma=self.conf.optim.gamma)
        return ({'optimizer': self.optim, 'lr_scheduler': self.scheduler})


    def get_resolution(self, epoch):
        assert conf.img.min_size <= conf.img.max_size
        if epoch <= conf.img.start_ramp:
            return conf.img.min_size
        if epoch >= conf.img.end_ramp:
            return conf.img.max_size

        # otherwise, linearly interpolate to the nearest multiple of 32
        interp = np.interp([epoch], [conf.img.start_ramp, conf.img.end_ramp], [conf.img.min_size, conf.img.max_size])
        final_size = int(np.round(interp[0] / 4)) * 4
        return final_size



if __name__ == "__main__":
    # wandb.login()
    conf = OmegaConf.load('config.yaml')
    experiment_root_dir = Path(conf.train.workdir)
    experiment_dir = experiment_root_dir / 'train' / f'exp_{datetime.now().strftime("%d-%b-%Y_%H_%M_%S")}'

    log_file = experiment_dir / f'log_{datetime.now().strftime("%d-%b-%Y_%H_%M_%S)")}.log'
    logger.opt(record=True).add(
        log_file,
        format="{message}",
        # format=' {time:YYYY-MMM HH:mm:ss} {name}:{function}:{line} <lvl>{message}</>',
        level=conf.log_level, rotation='5 MB'
    )

    logger.opt(colors=True).info(OmegaConf.to_yaml(conf))

    experiment_dir.mkdir(exist_ok=True)
    shutil.copy('config.yaml', Path(experiment_dir) / 'config.yaml')
    shutil.copy('train.py', Path(experiment_dir) / 'train.py')

    ffcv_device = f'cuda:{conf.train.gpu}'
    decoder_train = RandomResizedCropRGBImageDecoder(conf.img.size)
    decoder_val = RandomResizedCropRGBImageDecoder(conf.img.size)
    train_image_pipeline = [
        decoder_train,
        ImageMixup(alpha=0.3, same_lambda=False),
        RandomHorizontalFlip(),
        ToTensor(),
        ToDevice(torch.device(ffcv_device), non_blocking=True),
        ToTorchImage(),
        NormalizeImage(np.array(conf.img.mean), np.array(conf.img.std), np.float32)
    ]

    train_label_pipeline = [
        IntDecoder(),
        ToTensor(),
        Squeeze(),
        ToDevice(torch.device(ffcv_device), non_blocking=True)
    ]

    val_image_pipeline = [
        decoder_train,
        ToTensor(),
        ToDevice(torch.device(ffcv_device), non_blocking=True),
        ToTorchImage(),
        NormalizeImage(np.array(conf.img.mean), np.array(conf.img.std), np.float32)
    ]

    val_label_pipeline = [
        IntDecoder(),
        ToTensor(),
        Squeeze(),
        ToDevice(torch.device(ffcv_device), non_blocking=True)
    ]

    order = OrderOption.RANDOM if conf.train.distributed else OrderOption.QUASI_RANDOM
    train_dl = Loader(conf.train.dataset,
                          batch_size=conf.train.batch_size,
                          num_workers=conf.train.workers,
                          order=order,
                          os_cache=conf.train.in_memory,
                          drop_last=False,
                          pipelines={
                              'image': train_image_pipeline,
                              'label': train_label_pipeline
                          },
                          distributed=conf.train.distributed)

    val_dl = Loader(conf.val.dataset,
                        batch_size=conf.val.batch_size,
                        num_workers=conf.val.workers,
                        order=order,
                        os_cache=conf.val.in_memory,
                        drop_last=False,
                        pipelines={
                            'image': val_image_pipeline,
                            'label': val_label_pipeline
                        },
                        distributed=conf.train.distributed)


    # net = timm.create_model('cs3darknet_s', pretrained=True,num_classes=conf.train.num_classes)
    net = timm.create_model('efficientnet_b0', pretrained=True, num_classes=conf.train.num_classes)
    # criterion = torch.nn.CrossEntropyLoss(label_smoothing=conf.optim.label_smoothing)
    criterion = FocalLoss()
    optimizer = torch.optim.AdamW(net.parameters(),
                            lr=conf.train.lr,
                            weight_decay=conf.optim.weight_decay)


    model = PL_NET(net, optimizer, criterion, conf, train_dl, val_dl)
    model = model.to(ffcv_device)
    if not conf.train.restore_from_dir:
        logger.opt(colors=True).info('Loading pre-trained weights')
    # wandb_logger = WandbLogger(project="shai_ocr",
    #                            save_dir=experiment_dir,
    #                            offline=False,
    #                            name=conf.train.experiment_name
    #                            )
    # tsrb_logger = TensorBoardLogger(name="TimesFormer", save_dir=experiment_dir)
    dataModule = DataModule(train_dl, val_dl, conf)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    checkpoint_callback = ModelCheckpoint(dirpath=experiment_dir,
                                          mode='max',
                                          monitor='v_fb',
                                          save_top_k=1,
                                          every_n_epochs=1,
                                          filename='epoch_{epoch}_vfb_{v_fb:.4f}'
                                          )
    if not conf.train.restore_from_dir:
        logger.opt(colors=True).info('Training from scratch')

    else:
        logger.opt(colors=True).info('Loading checkpoint', )
        ckpt_file = list(Path(conf.train.restore_from_dir).glob('*.ckpt'))[0].as_posix()
    trainer = pl.Trainer(
        default_root_dir=experiment_dir,
        # logger=wandb_logger,
        max_epochs=conf.train.epochs,
        check_val_every_n_epoch= 1,
        accelerator='gpu',
        devices=1,
        accumulate_grad_batches=conf.optim.gradient_accumulations,
        gradient_clip_val=conf.train.gradient_clip,
        benchmark=True,
        callbacks=[checkpoint_callback, lr_monitor],
        auto_lr_find=True,
        auto_scale_batch_size='binsearch',
        precision=16,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
        enable_progress_bar=False,
        # val_check_interval=1,
        # fast_dev_run=True,
        # limit_train_batches=2,
        # limit_val_batches=2,
        # limit_test_batches=2
    )
    trainer.fit(model, dataModule)

    '''
    LR - finder option. In order to run it,
    comment trainer above and uncomment the lines below.
    '''
    # trainer = pl.Trainer(gpus=1, precision=16)
    # lr_finder = trainer.tuner.lr_find(model, min_lr=1e-7, max_lr=1e-1, num_training=100)
    # lr_finder.plot(suggest=True)
    # plt.xscale('log')
    # plt.show()

