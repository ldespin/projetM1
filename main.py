from model.FaceModel import Generator, Discriminator
from core.FaceGANNet import FaceGANNet
from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping
import torch.nn.functional as F
import os
import h5py as h5
import numpy as np


def get_imflow(flow, im_shape=(64, 64, 3)):
    hsv = np.zeros(im_shape, dtype=np.uint8)
    hsv[..., 1] = 255

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    im = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return im

def mse(output, target):
    return F.mse_loss(output, target)


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="checkpoint/", help="checkpoint path")

    parser.add_argument("--epoch", type=int, default=100, help="number of epoch")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--learning_rate_g", type=float, default=0.0001, help="generator learning rate")
    parser.add_argument("--learning_rate_d", type=float, default=0.0001, help="discriminator learning rate")

    parser.add_argument("--baseline", type =bool, default=False, help="baseline only")
    parser.add_argument("--data", type=str, default=None, help="data train path")
    parser.add_argument("--data_val", type=str, default=None, help="data val path")
    parser.add_argument("--data_test", type=str, default=None, help="data test cross val path")
    parser.add_argument("--modality", type=str, default=None, help="modality")

    hparams = parser.parse_args()

    checkpoint_callback = ModelCheckpoint(
            filepath=hparams.log_dir,
            verbose=True,
            monitor='val_loss',
            mode='min',
            prefix=''
        )

    early_stop_callback = EarlyStopping(
            monitor='val_loss',
            patience=30,
            verbose=False,
            mode='min'
        )

    data_mod = ['mouth_3', 'noOcclusion']
    metrics = {'mouth_3': [], 'noOcclusion': []}

    for modality in data_mod:
        hparams.modality = modality
        for rep in os.listdir('./data/'):
            for file in os.listdir('./data/'+rep):
                path = './data/'+rep+'/'+file
                f = h5.File(path, 'r')
                
                data_test = []
                if 'train' in file:
                    hparams.data = path

                elif 'val' in file:
                    hparams.data_val = path

                elif 'test' in file:
                    hparams.data_test = path


            generator = Generator()
            discriminator = Discriminator()
            net = FaceGANNet(hparams, generator, discriminator)
            trainer = pl.Trainer(max_epochs=hparams.epoch, checkpoint_callback=checkpoint_callback) 
            trainer.fit(net)
            print(trainer.test())