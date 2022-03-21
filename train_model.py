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
            actual =[]
            pred=[]
            
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
                    
                    #F_score takes sigmoid values, converts them to predections acc. to 0.5 threshold
                    #and return mean F-score and accuracy across
                    #all instances in the batch 
                    
                    fscore, accuracy=F_score(preds,labels)
                    
                    if (phase == "val"):
                        
                        labels = labels.cpu().numpy()
                        preds = preds.cpu().numpy()
                        actual.extend(labels)
                        pred.extend(preds)
                        
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
                pred = [round(num) for num in pred]
                
                cm = multilabel_confusion_matrix(actual,pred)
                labels = ["Foggy","Rainy","Snowy","Sunny","Cloudy","Clear","Wet"]
                disp = ConfusionMatrixDisplay(cm, display_labels=labels)
                disp.plot()
                plt.show()
                output_folder = './'
                plt.savefig(os.path.join(output_folder, 'resnet_cm_roaddata%s_epochs.png' % epoch))


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

