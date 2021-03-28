import torch.nn as nn
import torchvision.transforms as transforms
import sys
import torch
sys.path.insert(0, './reconstruction/flow')
from dataloaderFlow_h5 import *
print("import")
sys.path.insert(0, './utils')
import utils
import constantes
from numpy import genfromtxt
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import numpy as np
np.random.seed(0)
import h5py

class Interpolate(nn.Module):
    def __init__(self, size):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size

    def forward(self, x):
        x = self.interp(x, size=self.size)
        return x

img_transform = transforms.Compose([
    transforms.ToTensor(),
])

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # Upsampling
        self.down = nn.Sequential(
            nn.Conv2d(2, 8, 3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2)
            
            #"nouvelle couche"
            #nn.Conv2d(32, 64, 3, padding=1),
            #nn.ReLU(True),
            #nn.BatchNorm2d(64),
            #nn.MaxPool2d(2)
            )
        # Upsampling
        self.up = nn.Sequential(
            nn.Linear(8*8*32,256),
            nn.ReLU(True),
            #nouvelle couche
            #nn.Linear(512,256),
            #nn.ReLU(True),
            #fin
            nn.Linear(256,6)
            #nn.Softmax()
            )

    def forward(self, img):
        out = self.down(img)
        out = self.up(out.view(out.size(0), -1))
        return out

batch_size = 32
learning_rate = 1e-3
def train(num_epochs,datasets,train_it,saveFolder):
    #f = h5py.File(hparams.data, 'r')
    _,train_dir = utils.create_paths(datasets,["noOcclusion"])
    """print(train_dir)
    h5_mean = "/".join(train_dir[0].split("/")[:-1])+"/cnn_64_mean_std.h5"
    print(h5_mean)
    f = h5py.File(h5_mean,'r')
    mean_flow = img_transform(np.copy(f['mean']))
    std_flow = img_transform(np.copy(f['std']))
    f.close()"""
    cuda = True if torch.cuda.is_available() else False

    # Initialize generator and discriminator
    cnn = CNN()

    if cuda:
        cnn = cnn.cuda()
    dataset = FaceDataset(noisy_dir=train_dir,root_dir=train_dir,loading="apex",transform=img_transform)
    datasetLoad = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    # Optimizers

    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate, weight_decay=1e-5)

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    # ----------
    #  Training
    # ----------

    #step 1 : train Discriminator
    #print(repr(discriminator))
    sp = datasets[-1].split("/")
    pathLoadRoot = constantes.pathDatasets+ sp[0]+"/"+sp[1]+"/" +sp[0]+"_"+str(1+(train_it%10))
    pathLoadVal = pathLoadRoot + "_test/"
    pathLoadValNoOcc = pathLoadVal+"noOcclusion"
    flows = utils.load_h5(pathLoadValNoOcc,"apex")

    for epoch in range(num_epochs):
        mean_loss = 0
        for (_,_,img,label) in datasetLoad:
            img=img.float()
            #img = (img-mean_flow)/std_flow
            optimizer.zero_grad()
            img = Variable(img)
            if cuda:
                img = img.cuda()
            res = cnn(img)

            label = label.detach().clone().long()
            if cuda:
                label = label.cuda()
            CE_loss = nn.CrossEntropyLoss()(res, label)
            mean_loss += CE_loss.data.item()

            # ===================backward====================
            optimizer.zero_grad()
            CE_loss.backward()
            optimizer.step()
        mean_loss /= len(datasetLoad)
        #print("epoch : {}/{}, loss : {}".format(epoch,num_epochs,mean_loss))
        # =================validation=================
        acc = 0
        for (f,label) in flows:
            cnn.eval()
            f = np.asarray(f).squeeze()
            f = img_transform(f).float()
            f = f.unsqueeze(0)
            f = Variable(f)
            #f = (f-mean_flow)/std_flow
            if cuda:
                f = f.cuda()
            f_hat = cnn(f)
            f_hat = f_hat.detach().cpu().numpy()
            # print("predict : {}, label : {}".format(np.argmax(f_hat),label))
            if(np.argmax(f_hat) == label):
                acc +=1
        cnn.train()
        acc /= len(flows)
        #print("acc : {}".format(acc))
        if(epoch == 0 or acc >= max_acc):
            max_acc = acc
            best_model = cnn
        #print("max_acc : {}".format(max_acc))
    pathModel = saveFolder+"/cnn"+str(train_it)+".pth"
    torch.save(best_model.state_dict(), pathModel)


def test(dataset, occlusions, pathModel, train_it,read,occ,folder):
    cuda = True if torch.cuda.is_available() else False

    model = CNN()
    sp = dataset.split("/")
    """h5_mean = constantes.pathDatasets+sp[0]+"/newSplitTrainTestVal/"+sp[0]+"_"+str(train_it)+"_train/cnn_64_mean_std.h5"
    f = h5py.File(h5_mean,'r')
    mean_flow = img_transform(np.copy(f['mean']))
    std_flow = img_transform(np.copy(f['std']))
    f.close()
    """
    if cuda:
        model = model.cuda()
    checkpoint = torch.load(pathModel)
    model.load_state_dict(checkpoint)
    model.eval()
    pathRoot = "/".join(pathModel.split("/")[:-3])
    if(occ == "occ" and read == "read"):
        pathLoadRoot = folder+"/predictions/" +sp[0]+"_"+str(train_it)
        pathLoadTest = pathLoadRoot + "_test/"+occlusions[0]
    elif(occ == "occ"):
        pathLoadRoot = constantes.pathDatasets+sp[0]+"/"+sp[1]+"/"+sp[0]+"_"+str(train_it)
        pathLoadVal = pathLoadRoot+"_test/"
        pathLoadTest = pathLoadVal+occlusions[0]
    else:
        pathLoadRoot = constantes.pathDatasets+sp[0]+"/"+sp[1]+"/"+sp[0]+"_"+str(train_it)
        pathLoadVal = pathLoadRoot + "_test/"
        pathLoadTest = pathLoadVal+"noOcclusion"
    print("cnn test : "+pathLoadTest)
    flows = utils.load_h5(pathLoadTest,"apex")
    acc = 0
    with torch.no_grad():
        for (f,label) in flows:
            f = np.asarray(f).squeeze()
            f = img_transform(f).float()
            f = f.unsqueeze(0)
            f = Variable(f)
            #f = (f-mean_flow)/std_flow
            if cuda:
                f=f.cuda()
            f_hat = model(f)
            f_hat = f_hat.detach().cpu().numpy()
            # print("predict : {}, label : {}".format(np.argmax(f_hat),label))
            if(np.argmax(f_hat,axis=1) == label):
                acc +=1
        acc /= len(flows)
        #print("acc test : {}".format(acc))
    return acc
