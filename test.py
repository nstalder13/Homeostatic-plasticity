import pickle
import matplotlib.pyplot as plt
import numpy as np

from timeit import default_timer as timer
import torch
from torchvision import datasets, models, transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch import nn
import sys
import os
import torch.nn.functional as F
from torch import linalg as LA
import matplotlib.pyplot as plt
import pandas as pd
import datetime

class basic_net(nn.Module):
    def __init__(self):
        super(basic_net, self).__init__()
        self.lin1 = nn.Linear(784,100, bias = False)
        self.lin2 = nn.Linear(100,100, bias = False)
        self.lin3 = nn.Linear(100,10, bias = False)
        self.i = 0
        

    def forward(self, x):
        x = torch.flatten(x,1)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        x = F.log_softmax(x, dim=1)
        return x
    
    def weight_norm(self):
        with torch.no_grad():
            for param in self.parameters():
                if(len(param.shape) == 2):
                    F.normalize(param, dim = 1, out = param )
                    """
                    if(self.i % 200 == 0):
                        #print(LA.vector_norm(param, dim = 1)[0])
                        x = 0
                    self.i += 1
                    """
    
    def weight_norm_lay(self):
        with torch.no_grad():
            for param in self.parameters():
                if(len(param.shape) == 2):
                    F.normalize(param, dim = 0, out = param )

    def get_weight_mag(self):
        mag = 0
        i = 0
        for param in self.parameters():
                if(len(param.shape) == 2):
                    mag += torch.mean(torch.abs(param))
                    i+=1
        return mag.item()/i


def permut_MNIST(x,idx):
    #print(x.view(64,-1)[:,idx][0] == x[0].view(-1)[idx])  
    with torch.no_grad():  
        return x.view(x.shape[0],-1)[:,idx]
    

def run_ef03():
    #parameters which define, which dataset variation should be used, if both are true permuted mnist is run
    #for non true nothing runs
    permuted_mnist = False
    random_label_mnist = True

    #activates neuron wise weight normalization
    nw_weight_norm = True

    #choose your optimizer, if False Adam is used
    Sgd = False

    now = datetime.datetime.now()
    time = str(now.time())[0:8]
    epochs = 8000
    root = '.'

    if(permuted_mnist):
        epochs = 1000
        alpha = 2e-2
        path ="./Results/perm_"
        if(nw_weight_norm):
            path += "nwwn_" + time
        else:
            path += "basic_" + time

    elif(random_label_mnist):
        alpha = 3e-4
        epochs = 8000
        path ="./Results/randm_"
        if(nw_weight_norm):
            path += "nwwn_" + time
        else:
            path += "basic_" + time
    
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")
    print("permuted_mnist =", permuted_mnist, " randomlabel_mnist =", random_label_mnist, " nw_weight_norm =", nw_weight_norm, " SGD =", Sgd, " Adam =", not(Sgd), " Alpha =", alpha)

    #datasets/Model/Loss init:
    f_train_data = datasets.MNIST(root, download=True, train=True, transform = transforms.ToTensor())
    model = basic_net().to()
    loss_fn = nn.NLLLoss()

    if(Sgd):
        optimizer = torch.optim.SGD(model.parameters(),alpha)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def train(dataloader, model, loss_fn, optimizer, idx = 0, permut = False, nw_weight_norm = False):
        size = len(dataloader.dataset)
        model.train()
        test_loss, correct = 0, 0
        num_batches = len(dataloader)

        if(permut):
            idx = idx.to(device)

        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            #permut MNIST
            if(permut):
                X = permut_MNIST(X,idx)

            # Compute prediction error
            pred = model.forward(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            optimizer.step()
            
            if(nw_weight_norm):
                model.weight_norm()

            #model.weight_norm_lay()
            
            test_loss += loss_fn(pred, y).item()
            temp = (pred.argmax(1) == y).type(torch.float)
            correct += temp.sum().item()

            optimizer.zero_grad()

            if batch % 624 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                #print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        test_loss /= num_batches
        correct /= size
        
        #print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

        return correct*100
    
    idx = 0
    mask = 0
    train_dataloader = 0
    avg_onl_akk = []
    mag = []

    
    for t in range(epochs):
        
        if(permuted_mnist):
            mask = torch.randperm(len(f_train_data))[:10000]
            train_data = torch.utils.data.Subset(f_train_data, mask)
            train_dataloader = DataLoader(train_data, batch_size=16)
            idx = torch.randperm(28*28)
            print(f"\nTask {t+1}\n-------------------------------")
            avg_onl_akk.append(train(train_dataloader, model, loss_fn, optimizer, idx, permut=True, nw_weight_norm=nw_weight_norm))
            mag.append(model.get_weight_mag())
            print("Avg. online accuracy: ", round(avg_onl_akk[-1],1), '%' )
            print("Avg. weight magnitude: ", round(mag[-1],3) )

        elif(random_label_mnist):
            if(t % 400 == 0):
                if(t != 0):
                    avg_onl_akk[-1] /= 400
                    print(f"\nTask {t/400}\n-------------------------------")
                    print('Avg. online accuracy: ', round(avg_onl_akk[-1],1), '%')
                mask = torch.randperm(len(f_train_data))[:1200]
                train_data = torch.utils.data.Subset(f_train_data, mask)
                train_data.dataset.targets = torch.randint(0,9,(60000,))
                train_dataloader = DataLoader(train_data, batch_size=16)
                avg_onl_akk.append(0)
                mag.append(model.get_weight_mag())

            avg_onl_akk[-1] += train(train_dataloader, model, loss_fn, optimizer, nw_weight_norm=nw_weight_norm)
            

    if(permuted_mnist):
        avg_onl_akk = np.array(avg_onl_akk).reshape(100,-1).mean(-1)      
        mag = np.array(mag).reshape(100,-1).mean(-1)
    elif(random_label_mnist):
        avg_onl_akk[-1] /= 400

    #saving the data
    os.mkdir(path)

    m_df = pd.DataFrame(data=avg_onl_akk).T
    m_df.to_excel(excel_writer = path +"/avg_akk.xlsx")

    v_df = pd.DataFrame(data=mag).T
    v_df.to_excel(excel_writer = path +"/mag.xlsx")

    print(mag)
    print(avg_onl_akk)
    plt.plot(avg_onl_akk)
    plt.savefig(path + "/avg_aKK")
    plt.clf()
    plt.plot(mag)
    plt.savefig(path +"/mag")
        
run_ef03()

