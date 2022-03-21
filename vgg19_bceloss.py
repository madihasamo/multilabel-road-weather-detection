import torch
import torch.nn as nn
import torch.nn.functional as F
#from focal_loss.focal_loss import FocalLoss
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset,SubsetRandomSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, multilabel_confusion_matrix,ConfusionMatrixDisplay
import time

import os
import copy,csv
import statistics
from torch.utils.data import Dataset
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, csv, train, test):
        self.csv = csv
        self.train = train
        self.test = test
        self.image_names = self.csv[:]['subject_data']
        print(self.image_names)
        self.labels = np.array(self.csv.drop(['user_name', 'subject_ids','subject_data'], axis=1))
        self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=45),
                transforms.ToTensor(),
            ])
        
        
    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        img = Image.open(os.path.join('./HGVData/HGVData/images/', self.image_names[index]))
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = torch.from_numpy(self.labels[index])
        return img, label
        

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_epochs = 50


def F_score(output, label, threshold=0.5, beta=1): #Calculate the accuracy of the model
    prob = output > threshold
    label = label > threshold

    TP = (prob & label).sum(1).float()
    TN = ((~prob) & (~label)).sum(1).float()
    FP = (prob & (~label)).sum(1).float()
    FN = ((~prob) & label).sum(1).float()
    accuracy = (TP+TN) /(TP+TN+FP+FN)
    precision = torch.mean(TP / (TP + FP + 1e-12))
    recall = torch.mean(TP / (TP + FN + 1e-12))
    F2 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-12)
    return F2.mean(0),accuracy.mean(0)


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
 

# Decay LR by a factor of 0.1 every 7 epochs
#exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

#model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,num_epochs)


#KFOLD NEW IMPLEMENTATION STARTS FROM HERE. ABOVE IS SAME SO FAR
k=5 #num_fold=3
 
# Define the K-fold Cross Validator
    
splits=KFold(n_splits=k,shuffle=True,random_state=42)

data_dir='./data'

import pandas as pd 
import os

train_csv = pd.read_csv("./HGVData/HGVData/labels.csv")
print("done reading csv file")
train_data = ImageDataset(train_csv, train= True, test = False)

dataset = ImageDataset(train_csv, train= True, test = False)

folds_results={}
best_models = {}
training_average =0
fscore_avg = 0
val_avg=0
val_std = []
train_std=[]
fscore_std = []


#weights will be used when we perform weighted loss
#weights = [len(dataset)/4490,len(dataset)/1890,len(dataset)/1889,len(dataset)/8950]
print('this is total dataset length {}'.format(len(dataset)))
      


for fold, (train_idx,val_idx) in enumerate(splits.split(dataset)):

    print('Fold {}'.format(fold + 1))

    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    test_sampler = torch.utils.data.SubsetRandomSampler(val_idx)
    #train_loader = DataLoader(dataset, batch_size=32, sampler=train_sampler)
    #test_loader = DataLoader(dataset, batch_size=32, sampler=test_sampler)
    model_ft = models.vgg19(pretrained=True)


    feature_extract=True
    num_classes=7
    set_parameter_requires_grad(model_ft, feature_extract)
    #by default param.require grad is true which is fine if we are training from scratch and aren't just extracting feature
    
    
    num_ftrs = model_ft.classifier[6].in_features
    model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)

    #num_ftrs = model_ft.fc.in_features
    #model_ft.fc = nn.Linear(num_ftrs,num_classes) 
    model_ft = model_ft.to(device)
    
    #Focal loss is the type of dynamic loss that applies a modulating term to the cross entropy loss 
    #in order to focus learning on hard misclassified examples.
    
    #In this focal loss implementation, gamma focusses on assigning weights to hard examples however alpha parameter (i-e: weight)
    #focuses on assigning weights to each class for dealing with class imbalancement. weights is None by default. 
    #criterion = FocalLoss(gamma=5)
    
    #Below we have cross entropy loss with static weights 
    criterion = nn.BCELoss(reduction='none')
    
    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    print(model_ft)
   
    #so below we are chaning dataloaders and train / val datasetset everytime for each fold before calling train_model 
    dataloaders = {'train': torch.utils.data.DataLoader(dataset, batch_size=32,num_workers=4,
                                             sampler=train_sampler)}
    dataloaders['val'] = torch.utils.data.DataLoader(dataset, batch_size=32,num_workers=4,
                                             sampler = test_sampler)
    
    dataset_sizes = {'train':len(train_sampler)}
    dataset_sizes['val'] = len(test_sampler)

    print("dataset_sizes : {0}".format(dataset_sizes))
    
    best_models[fold],train_acc,best_val_acc, best_fscore = train_model(model_ft, criterion, optimizer_ft, dataloaders,num_epochs)
    val ='Fold_{0}'.format(fold+1)
    fold_results={val:{"train_acc":train_acc,"best_val_acc":best_val_acc,"best_fscore":best_fscore,"model":best_models[fold]}}
    training_average += fold_results[val]["train_acc"]
    #print('current training  result: {}'.format(fold_results["Fold_"+str(fold+1)]["train_acc"]))
    fscore_avg += fold_results[val]["best_fscore"]
    val_avg += fold_results["Fold_"+str(fold+1)]["best_val_acc"]
    
    #saving the best validation and corresponding training results in an array for standard deviation calculation
    val_std.append(fold_results["Fold_"+str(fold+1)]["best_val_acc"].item())
    train_std.append(fold_results["Fold_"+str(fold+1)]["train_acc"].item())
    fscore_std.append(fold_results["Fold_"+str(fold+1)]["bset_fscore"].item())
    
    
    #saving the current fold model that gave best validation accuracy among all epochs
    #save_path = f'./db/fold_model-{fold}.pth'
    #torch.save(best_models[fold].state_dict(), save_path)

training_average = training_average/ (k)
val_avg =val_avg/(k)
fscore_avg = fscore_avg / (k)
print("5 fold training accuracy average {}".format(training_average))
print("5 fold validation accuracy average {}".format(val_avg))
print("5 fold validation fscore average {}".format(fscore_avg))

val_std = statistics.stdev(val_std)
train_std = statistics.stdev(train_std)
fscore_std = statistics.stdev(fscore_std)
print("Standard deviation of validation accuracy {}".format(val_std))
print("Standard deviation of training accuracy {}".format(train_std))
print("Standard deviation of validation fscore {}".format(fscore_std))


### initialise the final model with the pretrained model structure###
final_model = models.vgg19()
num_ftrs = final_model.classifier[6].in_features
final_model.classifier[6] = nn.Linear(num_ftrs,num_classes)
final_dict = final_model.state_dict()
for j in final_dict.keys():
    final_dict[j] = torch.stack([best_models[i].state_dict()[j].float() for i in range(len(best_models))], 0).mean(0)
    final_model.load_state_dict(final_dict)

save_path_final = f'./vgg19_roaddata_bceloss-{k}Folds{num_epochs}epochs.pth'
torch.save(final_model.state_dict(), save_path_final)

