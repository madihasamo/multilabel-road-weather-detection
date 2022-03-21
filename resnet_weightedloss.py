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
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix,classification_report
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
        
        
class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2,reduction='mean'):
        super(FocalLoss, self).__init__(weight,reduction=reduction)
        self.gamma = gamma
        self.weight = weight #weight parameter will act as the alpha parameter to balance class weights
        
        
    def forward(self, input, target):
        patterns = target.shape[0]
        tot = 0
        for b in range(patterns):
            ce_loss = F.cross_entropy(input[b:b+1,], target[b:b+1],reduction=self.reduction,weight=self.weight)
            pt = torch.exp(-ce_loss)
            focal_loss = (1 - pt) ** self.gamma * ce_loss
            tot = tot + focal_loss
        return tot/patterns
        
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

def train_model(model, criterion, optimizer, dataloaders,num_epochs):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_hist =[]
    val_hist= []
    
    #weights = [len(dataset)/5521,len(dataset)/2934,len(dataset)/2924,len(dataset)/8950]
    #class_weights = torch.FloatTensor(weights).cuda()
    
    train_loss = []
    val_loss = []
    best_fscore = 0
    

    
    for epoch in range(num_epochs):
        
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
           
            running_loss = 0.0
            epoch_acc =0 
            epoch_fscore = 0 
            batches_accuracy = []
            batches_fscore = []
            
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                labels = labels.to(torch.float32)
                
                # zero the parameter gradients before updating the parmeters w.r.t gradients for every mini 
                #batch processing because for each batch parameters have already been updated w.r.t accumulated gradients of
                #that particular batch.  
                optimizer.zero_grad()
 
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    
                    #output from the model returns raw outputs before sigmoid 
                    #apply sigmoid activation to get all the outputs between 0 and 1
                    preds = torch.sigmoid(outputs)
                    
                    #loss function accepts sigmoid output and labels 
                    loss = criterion(preds, labels)
                    loss = (loss * class_weights).mean()
                    #F_score takes sigmoid values, converts them to predections acc. to 0.5 threshold
                    #and return mean F-score and accuracy across
                    #all instances in the batch 
                    
                    fscore, accuracy=F_score(preds,labels)
           
                    batches_fscore.append(fscore)
                    batches_accuracy.append(accuracy)
                    
                  
                    threshold = torch.tensor([0.55])
                    
                    ##the function F_score already converts preds according to threshold. 
                    #preds = (preds>threshold).float()*1
                   
                    #_, preds = torch.max(outputs, 1)
                    #preds = torch.flatten(preds)
                    #labels = torch.flatten(labels)
              

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        #loss.requires_grad = True 
                        loss.backward()
                        optimizer.step()

                # statistics
                
                running_loss += loss.item() * inputs.size(0)
                #running corrects work for multi-class not multi-label
                #running_corrects += torch.sum(preds == labels.data)
              
                

            
            #Use this when you want to change the decay rate with number of epochs. leave it for now
            #if phase == 'train':
            #    scheduler.step()
      
            #in each epoch entire dataset (i-e: size of dataloaders is considered)
            #this accuracy is also right because running corrects is added up for all mini batches until the size of our
            #dataset's particular phase (i-e : size of training or validation)
            if phase == 'train':
                 print(dataset_sizes[phase])
                 epoch_loss = running_loss / dataset_sizes[phase]  
                 epoch_fscore = torch.stack(batches_fscore).mean()
                 epoch_acc = torch.stack(batches_accuracy).mean()
                 train_hist.append(epoch_acc)
                 train_loss.append(epoch_loss)

                 
                
            if phase == 'val':
                 print(dataset_sizes[phase])
                 epoch_loss = running_loss / dataset_sizes[phase]   
                 epoch_fscore = torch.stack(batches_fscore).mean()
                 epoch_acc = torch.stack(batches_accuracy).mean()                 
                 val_hist.append(epoch_acc)
                 val_loss.append(epoch_loss)
                

            print('\n{} Loss: {:.4f}   and Accuracy: {:.4f} '.format(
                phase, epoch_loss,epoch_acc))
            
            
            
         
           
            # deep copy the model parameters at a given epoch where f-score is maximum 
            if phase == 'val' and epoch_fscore > best_fscore:
                best_epoch_index = epoch
                best_fscore = epoch_fscore
                best_model_wts = copy.deepcopy(model.state_dict())
                best_val_acc = epoch_acc
                co_train_acc = train_hist[epoch]


    time_elapsed = time.time() - since
   
                   
  
    epochs = range(num_epochs)
    plt.plot(epochs, train_loss, 'g', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    output_folder = './'
    plt.savefig(os.path.join(output_folder, 'resnet18_road_data%s_epochs.png' % num_epochs))



    # load best model weights
    model.load_state_dict(best_model_wts)
    return model,co_train_acc, best_val_acc, best_fscore



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


#weights will be used when we perform weighted loss
#weights = [len(dataset)/4490,len(dataset)/1890,len(dataset)/1889,len(dataset)/8950]
print('this is total dataset length {}'.format(len(dataset)))
      
#class_weights = torch.FloatTensor(weights).cuda()
weights = len(dataset)/ dataset.labels.sum(axis=0)
class_weights = torch.FloatTensor(weights).cuda()



for fold, (train_idx,val_idx) in enumerate(splits.split(dataset)):

    print('Fold {}'.format(fold + 1))

    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    test_sampler = torch.utils.data.SubsetRandomSampler(val_idx)
    #train_loader = DataLoader(dataset, batch_size=32, sampler=train_sampler)
    #test_loader = DataLoader(dataset, batch_size=32, sampler=test_sampler)
    model_ft = models.resnet18(pretrained=True)


    feature_extract=True
    num_classes=7
    set_parameter_requires_grad(model_ft, feature_extract)
    #by default param.require grad is true which is fine if we are training from scratch and aren't just extracting feature
    
    
    #num_ftrs = model_ft.classifier[6].in_features
    #model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)

    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs,num_classes) 
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
print("Standard deviation of validation accuracy {}".format(val_std))
print("Standard deviation of training accuracy {}".format(train_std))

### initialise the final model with the pretrained model structure###
final_model = models.resnet18()

num_ftrs = final_model.fc.in_features
final_model.fc = nn.Linear(num_ftrs,num_classes) 
    
#num_ftrs = final_model.classifier[6].in_features
#final_model.classifier[6] = nn.Linear(num_ftrs,num_classes)

final_dict = final_model.state_dict()
for j in final_dict.keys():
    final_dict[j] = torch.stack([best_models[i].state_dict()[j].float() for i in range(len(best_models))], 0).mean(0)
    final_model.load_state_dict(final_dict)

save_path_final = f'./resnet18_roaddata_weightedloss-{k}Folds{num_epochs}epochs.pth'
torch.save(final_model.state_dict(), save_path_final)



