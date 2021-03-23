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
import cv2
import torch
import matplotlib.pyplot as plt


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
        rep = os.listdir('./data/')[0]
        for file in os.listdir('./data/'+rep):
            path = './data/'+rep+'/'+file
            f = h5.File(path, 'r')

            if 'train' in file:
                hparams.data = path
            elif 'val' in file:
                hparams.data_val = path
            elif 'test' in file:
                hparams.data_test = path
                f = h5.File(hparams.data_test, 'r')
                data_original = np.copy(np.asarray(f['noOcclusion']).transpose(0, 3, 1, 2))
                data_test = np.copy(np.asarray(f[modality]).transpose(0, 3, 1, 2))

        

        generator = Generator()
        discriminator = Discriminator()
        net = FaceGANNet(hparams, generator, discriminator)
        trainer = pl.Trainer(max_epochs=hparams.epoch, checkpoint_callback=checkpoint_callback) 
        trainer.fit(net)
        trainer.test()

        for i in range(len(data_original)):
            original_im = get_imflow(np.transpose(data_original[i], (1, 2, 0)))
            occluded_im = get_imflow(np.transpose(data_test[i], (1, 2, 0)))
            output_gan = generator((torch.from_numpy(data_test[i]).float().unsqueeze(0)))
            reconstructed_im = get_imflow(np.transpose(output_gan.detach().numpy().squeeze(), (1, 2, 0)))

            cv2.imwrite('results/'+modality+'_original_'+str(i) + '.png', original_im)
            cv2.imwrite('results/'+modality+'_occluded_' + str(i) + '.png', occluded_im)
            cv2.imwrite('results/'+modality+'_reconstructed_' + str(i) + '.png', reconstructed_im)
        