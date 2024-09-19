from transformers import AutoTokenizer,  get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
import numpy as np
from tqdm.auto import tqdm
from utils import DebertaStylo, FakeNewsDataset, compute_metric
import itertools
import copy

for i in range(torch.cuda.device_count()):
   print(torch.cuda.get_device_properties(i))
   

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Defining the batch size and the number of epochs for training. We use a batch size of 64; however, due to memory issues on GPU, we 
# have to iterate through the data one by one instead of in batches. (i.e., batch size of 1)
batch_size = 1
epoch_num = 100

# Defining the search space for each hyperparameter
num_unfreezes = [1, 2, 3]
learning_rates = [ 5e-4, 5e-5, 1e-4]
weight_decays = [0.01,0.05, 0.1]
dropout_probs = [0.1, 0.2]
accumulation_step = 64

# Initialising the train, validation and test set
checkpoint = "microsoft/deberta-v3-base"
train_set = FakeNewsDataset("preprocessed_data/train.csv", "stylo")
valid_set = FakeNewsDataset("preprocessed_data/valid.csv", "stylo")
test_set = FakeNewsDataset("preprocessed_data/test.csv", "stylo")

# Iterating through each hyperparameter combination
for num_unfreeze, learning_rate, weight_decay, dropout_prob in itertools.product(num_unfreezes, learning_rates, weight_decays, dropout_probs):
    torch.manual_seed(2543673)
    file_name = f"experiments/unfreeze_{num_unfreeze}_lr_{learning_rate}_weight_decay_{weight_decay}_dropout_{dropout_prob}.pt"
    print(file_name)
    # Loading the pre-trained DeBERTa tokeniser
    tokeniser = AutoTokenizer.from_pretrained(checkpoint)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    # Initialsing the DeBERTa Stylo model with the given dropout probability and device
    model = DebertaStylo(dropout_prob, device)
    print(model.model)
    optimiser = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr = learning_rate, weight_decay = weight_decay)
    total_training_steps = len(train_loader) * epoch_num
    scheduler = get_linear_schedule_with_warmup(optimizer=optimiser , num_warmup_steps=0, num_training_steps= total_training_steps)

    # Freezing all layers in our model
    for param in model.model.deberta.parameters():
        param.requires_grad = False
        
    num_layers = len(model.model.deberta.encoder.layer)
    
    # Unfreeze the last num_unfreeze layers of the pre-trained DeBERTa
    for layer in model.model.deberta.encoder.layer[num_layers-num_unfreeze:]:
        for param in layer.parameters():
            param.requires_grad = True
            
    # Unfreeze the parameters in the classification head and pooler
    for param in itertools.chain(model.model.classifier.parameters(), model.model.pooler.parameters()):
        param.requires_grad = True
        
     # Dictionary to store the training/validation metrics for the entire training
    train_stats = {"train_loss" : [], "valid_loss" : [], "train_accuracy" : [],  "valid_accuracy" : [], "train_p" : [], "valid_p" : [] , "train_r" : [], "valid_r" : [],
                    "train_f1" : [], "valid_f1" : []}

    # Storing the number of epochs elapsed where the validation loss has not improved beyond the current best
    epoch_no_improve = 0
    # Storing the minimum validation loss achieved when training
    min_val_loss = float('inf')
    # Stores the patience of early stopping
    early_stopping_epoch = 5
    # Stores the epoch where the model achieved the best validation loss
    best_epoch = 0
    # Stores the parameters of the model at the point where it achieved the best validation loss
    best_state_dict = None
    
    # Iterating through each epoch to train the model
    for epoch in range(epoch_num):
        # Dictionary to store the training/validation metrics for this epoch
        current_epoch_stats = {"train_loss" : [], "valid_loss" : [], "train_accuracy" : [],  "valid_accuracy" : [], "train_p" : [], "valid_p" : [] , "train_r" : [], "valid_r" : [],
                    "train_f1" : [], "valid_f1" : []}
        model.train()
        
        # Stores the predicted author of the training articles  
        y_pred_train = []
        # Stores the true author of the training articles
        y_true_train = []
        # Stores the training loss
        train_loss = 0
        
        with tqdm(range(len(train_loader))) as ptrain_bar :
            # Iterating through each batch of the training articles
            for batch_idx, (X_train_text, X_train_stylo, y_train) in enumerate(train_loader):
                # Passing the articles embeddings and stylometric features to the model to compute the logit scores for each article
                outputs = model(X_train_text,X_train_stylo, tokeniser).to("cpu")
                # Computing the cross-entropy loss over the batch of articles
                example_loss = F.cross_entropy(outputs, y_train)
                
                # Record the predicted author for each article in the batch
                y_pred_train.append(torch.argmax(outputs, dim = 1).item())
                # Record the correct  author for each article in the batch
                y_true_train.append(torch.argmax(y_train, dim = 1).item())
                # Aggregate the cross-entropy loss over the batch to the training loss
                train_loss += example_loss
                
                 # When we have iterated through 64 articles, we then perform backpropagation and record the training metrics
                if (batch_idx + 1) % accumulation_step == 0 or (batch_idx + 1 == len(train_loader)):
                    if (batch_idx + 1) == len(train_loader) and (batch_idx + 1) % accumulation_step != 0:
                        num_samples = len(train_loader) % accumulation_step
                    else:
                        num_samples = accumulation_step
                    
                    # Dividing the aggregated training loss by the number of samples in the current batch
                    train_loss /= num_samples
                    # Performing backpropgation on our model's parameters
                    train_loss.backward()
                    optimiser.step()
                    optimiser.zero_grad()
                    
                    # Computing the training accuracy,precision , recall and F1-score for the current batch
                    train_accuracy, train_p, train_r, train_f1 = compute_metric(y_pred_train, y_true_train)
                    
                    # Recording the batch accuracy, precision , recall and F1-score
                    current_epoch_stats['train_loss'].append(train_loss.detach().item())
                    current_epoch_stats['train_accuracy'].append(train_accuracy)
                    current_epoch_stats['train_p'].append(train_p)
                    current_epoch_stats['train_r'].append(train_r)
                    current_epoch_stats['train_f1'].append(train_f1)
                    
                    # Resetting the variables for the next training batch 
                    y_pred_train = []
                    y_true_train = []
                    train_loss = 0
                    
                scheduler.step()
            
                ptrain_bar.update(1)
                

        model.eval()
        # Stores the predicted author of the validation articles 
        y_pred_val = []
        # Stores the true author of the validation articles  
        y_true_val = []
        # Storing the validation loss
        val_loss = 0
        
        with torch.no_grad(),tqdm(range(len(valid_loader))) as pvalid_bar:
            # Iterating through each batch of the validation articles
            for batch_idx, (X_valid_text,X_valid_stylo, y_valid) in enumerate(valid_loader):
                # Passing the articles embeddings and stylometric features to the model to compute the logit scores for each article
                outputs = model(X_valid_text,X_valid_stylo,  tokeniser).to("cpu")
                # Computing the cross-entropy loss over the batch of articles
                example_loss = F.cross_entropy(outputs, y_valid)
                
                # Record the predicted author for each article in the batch
                y_pred_val.append(torch.argmax(outputs, dim = 1).item())
                # Record the correct  author for each article in the batch
                y_true_val.append(torch.argmax(y_valid, dim = 1).item())
                # Aggregate the cross-entropy loss over the batch to the validation loss
                val_loss += example_loss.detach().item()
                
                # When we have iterated through 64 articles, we then record the validation metrics
                if (batch_idx + 1) % accumulation_step == 0 or (batch_idx + 1 == len(valid_loader)): 
                    
                    if batch_idx + 1 == len(valid_loader) and (batch_idx + 1) % accumulation_step != 0 :
                        num_samples = len(valid_loader) % accumulation_step
                    else:
                        num_samples = accumulation_step
        
                    # Dividing the aggregated validation loss by the number of samples in the current batch
                    val_loss /= num_samples
                    # Computing the validation accuracy,precision , recall and F1-score for the current batch      
                    val_accuracy, val_p, val_r, val_f1 = compute_metric(y_pred_val, y_true_val)

                    # Recording the batch accuracy, precision , recall and F1-score
                    current_epoch_stats['valid_loss'].append(val_loss)
                    current_epoch_stats['valid_accuracy'].append(val_accuracy)
                    current_epoch_stats['valid_p'].append(val_p)
                    current_epoch_stats['valid_r'].append(val_r)
                    current_epoch_stats['valid_f1'].append(val_f1)
                    
                    # Resetting the variables for the next validation batch  
                    y_pred_val = []
                    y_true_val = []
                    val_loss = 0


                pvalid_bar.update(1)
            
        # Computing the mean training/validation metrics over all batches for the current epoch 
        for metric, values in current_epoch_stats.items():
            epoch_mean = np.mean(values)
            train_stats[metric].append(epoch_mean)
        
        # Obtaining the mean training/validation metric for the current epoch 
        epoch_train_loss = train_stats["train_loss"][-1]
        epoch_val_loss = train_stats["valid_loss"][-1]
        epoch_train_accuracy = train_stats["train_accuracy"][-1]
        epoch_val_accuracy = train_stats["valid_accuracy"][-1]
        epoch_train_f1 = train_stats["train_f1"][-1]
        epoch_val_f1 = train_stats["valid_f1"][-1]
        
        print("Epoch {:d} - train_loss: {:.4f}, train_accuracy: {:.4f},train_f1: {:.4f}, valid_loss: {:.4f}, valid_accuracy: {:.4f},valid_f1: {:.4f}".format(epoch,epoch_train_loss, epoch_train_accuracy, epoch_train_f1, epoch_val_loss, epoch_val_accuracy , epoch_val_f1))
        
         # Checking if the current validation loss is smaller than the minimum validation loss achieved by previous epochs
        if epoch_val_loss < min_val_loss:
            # If condition is satisfied, we record the current validation loss as the new minimum
            min_val_loss = epoch_val_loss
            # Reset the no-improvement epochs elapsed to 0
            epoch_no_improve = 0
            # Record this current epoch as the one where the best validation loss is achieved
            best_epoch = epoch
            # Record the parameter of our model at the current epoch
            best_state_dict = copy.deepcopy(model.state_dict())    
        else:
            # If condition is not satisfied, we increment the no-improvement epochs elapsed
            epoch_no_improve += 1
            
            # If no-improvement epochs elapsed is greater than the patience, we terminate training early to avoid overfitting
            if epoch_no_improve >= early_stopping_epoch:
                break
    
    # Loading our model with the parameters at the epoch where the best validation loss was achieved
    model.load_state_dict(best_state_dict)    
    model.eval()
    # Dictionary to store the test metrics for the baseline
    test_stats = {"test_loss" : [], "test_accuracy" : [],  "test_p" : [],  "test_r" : [], "test_f1" : []}
    # Stores the predicted author of the test articles
    y_pred_test = []
    # Stores the true author of the test articles  
    y_true_test = []
    # Storing the test loss
    test_loss = 0

    with torch.no_grad():
        # Iterating through each batch of the test articles
        for batch_idx, (X_test_text,X_test_stylo,  y_test) in enumerate(test_loader):
            # Passing the articles embeddings and stylometric features to the model to compute the logit scores for each article
            outputs = model(X_test_text,X_test_stylo,tokeniser).to("cpu")
            # Computing the cross-entropy loss over the batch of articles
            example_loss = F.cross_entropy(outputs, y_test)
            
            # Record the predicted author for each article in the batch
            y_pred_test.append(torch.argmax(outputs, dim = 1).item())
            # Record the correct  author for each article in the batch
            y_true_test.append(torch.argmax(y_test, dim = 1).item())
            # Aggregate the cross-entropy loss over the batch to the test loss
            test_loss += example_loss.detach().item()
            
            # When we have iterated through 64 articles, we then record the test metrics
            if (batch_idx + 1) % accumulation_step == 0 or (batch_idx + 1 == len(test_loader)):          
                
                if batch_idx + 1 == len(test_loader) and (batch_idx + 1) % accumulation_step != 0:
                    num_samples= len(test_loader) % accumulation_step
                else:
                    num_samples = accumulation_step
                    
                # Dividing the aggregated test loss by the number of samples in the current batch
                test_loss /= num_samples
                # Computing the test accuracy,precision , recall and F1-score for the current batch
                test_accuracy, test_p, test_r, test_f1 = compute_metric(y_pred_test, y_true_test)
                
                # Recording the batch accuracy, precision , recall and F1-score
                test_stats['test_loss'].append(test_loss)
                test_stats['test_accuracy'].append(test_accuracy)
                test_stats['test_p'].append(test_p)
                test_stats['test_r'].append(test_r)
                test_stats['test_f1'].append(test_f1)
                
                # Resetting the variables for the next test batch 
                y_pred_test = []
                y_true_test = []
                test_loss = 0

    # Computing the mean test metrics over all batches
    for metric, values in test_stats.items():
        batch_mean = np.mean(values)
        test_stats[metric] = batch_mean

    test_loss = test_stats["test_loss"]
    test_accuracy = test_stats["test_accuracy"]
    test_f1 = test_stats["test_f1"]
    print("test_loss: {:.4f}, test_accuracy: {:.4f},test_f1: {:.4f}".format(test_loss, test_accuracy, test_f1))
    # Recording the training, validation and test metrics of our model, as well as the epoch and parameters where it achieved the best validation loss
    model_info = {"train_stats" : train_stats, "test_stats" : test_stats, "model_params" : best_state_dict, "best_epoch" : best_epoch}
    torch.save(model_info,file_name)