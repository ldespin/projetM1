import torch.nn as nn
from model.DNNBlock import DNNBlock
import torch

class Generator(nn.Module):
    def __init__(self, input_shape=(2, 64, 64), output_shape=(2, 64, 64), *args, **kwargs):
        super(Generator, self).__init__()

        self.input_shape = input_shape
        self.output_shape = output_shape

        self.enc = nn.Sequential(DNNBlock.conv2d_block(self.input_shape[0], 64, kernel_size=3, padding=1),
            DNNBlock.conv2d_block(64, 128, kernel_size=3, padding=1),
            DNNBlock.conv2d_block(128, 256, kernel_size=3, padding=1),
            DNNBlock.conv2d_block(256, 256, kernel_size=3, padding=1),
            DNNBlock.conv2d_block(256, 512, kernel_size=3, padding=1, pooling=False))

        self.dec = nn.Sequential(DNNBlock.conv2d_block(512, 256, kernel_size=3, padding=1, pooling=False),
            DNNBlock.conv2d_block(256, 256, kernel_size=3, padding=1, pooling=False, upsample=True),
            DNNBlock.conv2d_block(256, 128, kernel_size=3, padding=1, pooling=False, upsample=True),
            DNNBlock.conv2d_block(128, 64, kernel_size=3, padding=1, pooling=False, upsample=True),
            DNNBlock.conv2d_block(64, self.output_shape[0], kernel_size=1, activation='none', batch_norm=False, padding=0, pooling=False, upsample=True))

    def forward(self, x):
        e_out = self.enc(x)
        d_out = self.dec(e_out)

        return d_out

class Discriminator(nn.Module):
    def __init__(self, input_shape=(64, 64, 2), output_shape=(1,), *args, **kwargs):
        super(Discriminator, self).__init__()

        self.input_shape = input_shape
        self.output_shape = output_shape

        self.enc = nn.Sequential(DNNBlock.conv2d_block(self.input_shape[-1], 64, kernel_size=3, padding=1),
            DNNBlock.conv2d_block(64, 128, kernel_size=3, padding=1),
            DNNBlock.conv2d_block(128, 256, kernel_size=3, padding=1),
            DNNBlock.conv2d_block(256, 256, kernel_size=3, padding=1),
            DNNBlock.conv2d_block(256, 512, kernel_size=3, padding=1, pooling=False))

        self.classifier = nn.Sequential(nn.Flatten(),
            DNNBlock.fc_block((self.input_shape[0] // 2**4) * (self.input_shape[1] // 2**4) * 512, 512, dropout=0.2),
            DNNBlock.fc_block(512, self.output_shape[-1], activation='sigmoid', batch_norm=None))

    def forward(self, x):
        e_out = self.enc(x)
        c_out = self.classifier(e_out)

        return c_out