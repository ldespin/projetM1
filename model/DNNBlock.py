import torch.nn as nn

class DNNBlock:
    @staticmethod
    def conv2d_block(in_channel, out_channel, activation='elu', batch_norm=True, pooling=True, upsample=False, dropout=None, *args, **kwargs):

        activations = nn.ModuleDict([
                    ['lrelu', nn.LeakyReLU()],
                    ['relu', nn.ReLU()],
                    ['elu', nn.ELU()],
                    ['sigmoid', nn.Sigmoid()],
                    ['none', nn.Identity()]
        ])
        
        layers = [nn.Conv2d(in_channel, out_channel, *args, **kwargs)]

        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if activation != None:
            layers.append(activations[activation])
        if pooling:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        if upsample:
            layers.insert(0, nn.Upsample(scale_factor=2, mode='nearest'))
        if dropout != None:
            layers.append(nn.Dropout2d(p=dropout))

        return nn.Sequential(*layers)

    @staticmethod
    def fc_block(in_feature, out_feature, activation='elu', batch_norm=True, dropout=None, *args, **kwargs):

        activations = nn.ModuleDict([
                    ['lrelu', nn.LeakyReLU()],
                    ['relu', nn.ReLU()],
                    ['elu', nn.ELU()],
                    ['sigmoid', nn.Sigmoid()],
                    ['none', nn.Identity()]
        ])
        
        layers = [nn.Linear(in_feature, out_feature, *args, **kwargs)]

        if batch_norm:
            layers.append(nn.BatchNorm1d(out_feature))
        if activation != None:
            layers.append(activations[activation])
        if dropout != None:
            layers.append(nn.Dropout(p=dropout))

        return nn.Sequential(*layers)