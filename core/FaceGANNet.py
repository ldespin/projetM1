import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchsummary import summary
import torch
import torch.nn.functional as F
import torchvision
from core.FaceDataset import FaceDataset
from collections import OrderedDict

class FaceGANNet(pl.LightningModule):

    def __init__(self, hparams, generator, discriminator, exp_rec):
        super(FaceGANNet, self).__init__()

        self.generator = generator
        self.discriminator = discriminator
        self.hparams = hparams

    def on_save_checkpoint(self, checkpoint):
        torch.save(self.generator.state_dict(), self.hparams.log_dir + 'generator_test.pth')

    def forward(self,x):
        return self.generator(x)

    def mse(self,y_hat,y):
        return F.mse_loss(y,y_hat)
    
    def adversarial_loss(self, output, target):
        target = target.to(torch.float32)
        return F.binary_cross_entropy(output, target)
    
    def accuracy(self, y_hat, y):
        predicted = y_hat >= 0.5
        total = y.size(0)
        correct = (predicted == y).sum().item()

        return 100 * (correct / total)

    def maccuracy(self, output, target):
        prediction = output.argmax(dim=1)

        corrects = (prediction == target)
        accuracy = corrects.sum().float() / float(target.size(0))
        return 100 * accuracy


    def training_step(self, batch, batch_idx, optimizer_idx=0):
        # batch returns x and y tensors
        input_images, output_images, _ = batch.values()

        
        # As there are 2 optimizers we have to train for both using 'optimizer_idx'
        ## Generator
        
        if optimizer_idx == 0:
            self.reconstruction = self.generator(input_images) 

            valid = torch.full((output_images.size(0),1),0.1)   

            # Calculating generator loss
            # How well the generator can create real images

            if self.hparams.baseline:
                g_loss = self.mse(self.reconstruction, output_images)

            else:
                g_loss = self.mse(self.reconstruction, output_images)+self.adversarial_loss(self.discriminator(self.reconstruction),valid)
            
            # for output and logging purposes (return as dictionaries)
            tqdm_dict = {'g_loss': g_loss}
            output = OrderedDict({
                'loss': g_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict,
                'g_loss': g_loss
            })
            return output

        ## Discriminator
        if optimizer_idx == 1:
            # Calculating disciminator loss
            # How well discriminator identifies the real and fake images
            valid = torch.full((output_images.size(0),1),0.1)
            real_output = self.discriminator(output_images)
            real_loss = self.adversarial_loss(real_output, valid)
            real_acc = self.accuracy(real_output, valid)
            
            fake = torch.full((input_images.size(0),1),0.9)
            fake_output = self.discriminator(self.reconstruction.detach())
            fake_loss = self.adversarial_loss(fake_output, fake)
            fake_acc = self.accuracy(fake_output, fake)

            d_loss = (real_loss + fake_loss)/2.0
            d_acc = (real_acc + fake_acc) / 2

            # for output and logging purposes (return as dictionaries)
            tqdm_dict = {'d_loss': d_loss,'d_loss_real': real_loss,'d_loss_fake': fake_loss, 'd_acc': d_acc,'d_acc_real': real_acc,'d_acc_fake': fake_acc,}
            output = OrderedDict({
                'loss': d_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict,
                'd_loss': d_loss,
                'd_loss_real': real_loss,
                'd_loss_fake': fake_loss,
                'd_acc': d_acc
            })
            return output

    def configure_optimizers(self):
        if self.hparams.baseline :
            optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.hparams.learning_rate_g, betas=(0.4, 0.999))
            return [optimizer_G], []
        else:
            optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.hparams.learning_rate_g, betas=(0.4, 0.999))
            optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.hparams.learning_rate_d, betas=(0.4, 0.999))
            # return the list of optimizers and second empty list is for schedulers (if any)
            return [optimizer_G, optimizer_D], []

    def test_step(self, batch, batch_idx):
        of_input, of_output, label = batch.values()

        of_reconstruction = self.forward(of_input)
        test_loss = self.mse(of_reconstruction, of_output)

        prediction = self.exp_rec(of_reconstruction)
        test_acc_r = self.maccuracy(prediction, label)

        prediction = self.exp_rec(of_output)
        test_acc_o = self.maccuracy(prediction, label)

        return {'test_loss': test_loss, 'test_acc_r': test_acc_r, 'test_acc_o': test_acc_o}

    def test_end(self, outputs):
        avg_test_loss = torch.stack([x['test_loss'] for x in outputs]).mean()

        log_dict = {'avg_test_loss': avg_test_loss}

        return {'test_loss': avg_test_loss, 'log': log_dict}

    def validation_step(self, batch, batch_idx):
        of_input, of_output, _ = batch.values()

        of_reconstruction = self.forward(of_input)
        val_loss = self.mse(of_reconstruction, of_output)

        return {'val_loss': val_loss}

    def validation_end(self, outputs):
        avg_val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        log_dict = {'avg_val_loss': avg_val_loss}

        return {'val_loss': avg_val_loss, 'log': log_dict}

    def train_dataloader(self):
        dataset = FaceDataset(self.hparams.data, 'train', self.hparams.modality)
        return DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=0)

    def test_dataloader(self):
        dataset = FaceDataset(self.hparams.data_test, 'test', self.hparams.modality)
        return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    def val_dataloader(self):
        dataset = FaceDataset(self.hparams.data_val, 'val', self.hparams.modality,)
        return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)