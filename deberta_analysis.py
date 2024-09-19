import matplotlib.pyplot as plt
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from collections import OrderedDict
from utils import FakeNewsDataset, DebertaStylo, create_batch
import shap
import pickle
import numpy as np
import pandas as pd


def find_best_model():
    """
    
    This function finds the hyperparameter combination of DeBERTa Stylo that achieved the 
    highest validation accuracy. After this, it plots the training performance of the model with the
    given hyperparameter configuration. Training performance includes the loss, accuracy and F1-score 
    on the training and validation set.

    Returns:
        OrderedDict : The parameters of our model with the best hyperparameter configuration
    """
    # Storing the best validation accuracy achieved by the hyperparameters
    best_accuracy = 0
    # Storing the metadata of the model with the best hyperparameter configuration
    best_dict = None
    # Storing the hyperparameter configuration that achieved the best validation accuracy
    best_hyperparameter = None

    # Iterating through the training performance of each hyperparameter configuration
    for filename in os.listdir("experiments"):
        if filename == "baselines" : continue
        model_dict = torch.load(f"experiments/{filename}")
        # Retrieving the epoch where the model achieved the lowest validation loss with the current configuration 
        best_epoch = model_dict['best_epoch']
        # Obtaining the validation accuracy of the model with the current configuration at epoch where it achieved the lowest validation loss
        val_accuracy = model_dict['train_stats']['valid_accuracy'][best_epoch]
        print(f"Hyperparameter : {filename}, Validation Accuracy : {val_accuracy}\n\n")
        
        # Determining if the validation accuracy of the current configuration is higher than the ones found before
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_dict = model_dict
            best_hyperparameter = filename
        
    print(f"Best Hyperparameters : {best_hyperparameter}, Best Model Train Stats : {best_dict['train_stats']}, Best Model Test Stats : {best_dict['test_stats']}")
    train_stats = best_dict['train_stats']
    
    # Obtaining the training and validation loss of the best-performing model
    train_loss , val_loss = train_stats['train_loss'], train_stats['valid_loss']
    # Plotting the training and validation loss
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label ='Valid Loss')
    plt.xlabel("Epoch", fontsize=16)
    plt.ylabel("Loss", fontsize=16)
    plt.xticks(np.arange(len(train_loss)), np.arange(1, len(train_loss)+1), fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.savefig("best_model_result/model_loss_plot.png")
    plt.close()
    
    # Obtaining the training and validation accuracy of the best-performing model
    train_accuracy , val_accuracy = train_stats['train_accuracy'], train_stats['valid_accuracy']
    # Plotting the training and validation accuracy against epoch
    plt.plot(train_accuracy, label='Train Accuracy')
    plt.plot(val_accuracy, label ='Valid Accuracy')
    plt.xlabel("Epoch", fontsize=16)
    plt.ylabel("Accuracy",fontsize=16)
    plt.xticks(np.arange(len(train_accuracy)), np.arange(1, len(train_accuracy)+1), fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)   
    plt.tight_layout()
    plt.savefig("best_model_result/model_accuracy_plot.png")
    plt.close()
    
    # Obtaining the training and validation F1-score of the best-performing model
    train_f1 , val_f1 = train_stats['train_f1'], train_stats['valid_f1']
    # Plotting the training and validation F1-score against epoch
    plt.plot(train_f1, label='Train F1')
    plt.plot(val_f1, label ='Valid F1')
    plt.xlabel("Epoch", fontsize=16)
    plt.ylabel("F1 Score", fontsize=16)
    plt.xticks(np.arange(len(train_f1)), np.arange(1, len(train_f1)+1), fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.savefig("best_model_result/model_f1_plot.png")
    plt.close()
    
    
    return best_dict['model_params']


def plot_bar(shap_values):
    """
    
    Given the SHAP values of the linguistic features, this function plots a bar chart of the top 
    5 most important feature the discimrinate a given author's writing style from others.
    
    Args:
        shap_values (np.array): A numpy array containing the SHAP value for each feature/author pair
    """

    bar_width = 0.25
    models_triplet = [[4,6, 2], [3, 5, 1], [0, 7, 8]]
    
    for models in models_triplet:
        tick_labels = []
        tick_position = []
        importances = []
        
        for i, label in enumerate(models):
            # Computing the global importance of each linguistic feature for the given author
            global_fi = shap_values[:, :, label].mean(0)
            features_names = np.array(['semantic'] + stylo_columns)
            # Aggregating the global importance of the [CLS] embeddings to compute an overall global importance for the text's content
            text_importance = np.sum(global_fi[:768])
            importance = np.insert(np.delete(global_fi, range(768), 0), 0, text_importance)
            
            # Retrieving the top 5 features with the highest global importance for the given author
            max_idx = np.argsort(importance)[-5:]
            features_names = features_names[max_idx]
            x = np.arange(len(features_names))
            
            # Retrieving the importance value of the top 5 features with the highest global importance
            importance = importance[max_idx]
            # Plot a bar chart of the top 5 most important features
            plt.bar(x + i * bar_width ,importance, bar_width ,label=label_mapping[label])
            
                
            tick_position += list(x + i * bar_width)
            tick_labels.append(list(features_names))
            importances.append(list(importance))

        importances = [item for sublist in zip(*importances) for item in sublist]  
        tick_labels = [item for sublist in zip(*tick_labels) for item in sublist]  
        tick_position.sort()    
        

        # Adding the importance value to top of the bar to improve interpretability
        for idx in range(len(tick_position)):
            plt.text(tick_position[idx], importances[idx] ,round(importances[idx],4), ha = 'center', fontsize=11, rotation =90)
        
        plt.yticks(fontsize=12)
        plt.xticks( ticks=tick_position, labels=tick_labels, rotation=35, ha='right', fontsize=12)
        plt.ylabel("Global Feature Importance", fontsize=12)
        plt.xlabel("Features", fontsize=12)
        plt.legend()
        plt.tight_layout()
        plt.show()
            
        plt.close()
    
def stylometric_analysis(stylo_params):
    """
    
    For each news article in the test set, this function applies SHAP to our best-performing model 
    to compute the SHAP value for each feature/author pair. These values provide a local explanation for 
    the instance.
    

    Args:
        stylo_params (OrderedDict) : The parameters of our model with the best hyperparameter configuration
    """
    # Initialising the DeBERTa Stylo model and its corresponding tokeniser
    tokeniser = AutoTokenizer.from_pretrained(checkpoint)
    model = DebertaStylo(0.1, device)
    # Loading the parameters of the model with the best hyperparameter configuration to our initialised model
    model.load_state_dict(stylo_params)
    model.eval()
    
    batch_size = 2
    # Randomly sample 1,000 training examples for feature integration
    train_set.X = train_set.X.sample(1000, random_state=2543673)
    #test_set.X = test_set.X.sample(4, random_state=2543673)
    with torch.no_grad():
        train_features = None
        test_features = None
        
        # Iterating through the articles from the training set
        for idx in range(0, len(train_set.X), batch_size):
            print("Train idx : ", idx)
            # Obtaining the text of the articles
            X_text = train_set.X.iloc[idx:idx+batch_size]["text"].values.tolist()
            # Obtaining the stylometric features of the articles
            X_stylo = torch.tensor(train_set.X.iloc[idx:idx+batch_size][stylo_columns].values, dtype=torch.float32)
            # Encoding the text of the articles into embeddings
            batch_data = create_batch(X_text, tokeniser, device, stylo=True)
            # Retrieve the [CLS] output embedding of the article
            embeddings = model.model.pooler(model.model.deberta(**batch_data).last_hidden_state).cpu()
            
            # Concatenating the [CLS] embedding with the stylometric features
            concat_features = torch.cat((embeddings, X_stylo), dim=-1)
            # Appending the concatenated vectors to the training features
            train_features = torch.cat((train_features, concat_features), dim = 0) if train_features is not None else concat_features
        
        # Iterating through the articles from the test set
        for idx in range(0, len(test_set.X), batch_size):
            print("Test idx : ", idx)
            # Obtaining the text of the articles
            X_text = test_set.X.iloc[idx:idx+batch_size]["text"].values.tolist()
            # Obtaining the stylometric features of the articles
            X_stylo = torch.tensor(test_set.X.iloc[idx:idx+batch_size][stylo_columns].values, dtype=torch.float32)
            # Encoding the text of the articles into embeddings
            batch_data = create_batch(X_text, tokeniser, device, stylo=True)
            # Retrieve the [CLS] output embedding of the article
            embeddings = model.model.pooler(model.model.deberta(**batch_data).last_hidden_state).cpu()
            
            # Concatenating the [CLS] embedding with the stylometric features
            concat_features = torch.cat((embeddings, X_stylo), dim=-1)
            # Appending the concatenated vectors to the test features
            test_features = torch.cat((test_features, concat_features), dim = 0) if test_features is not None else concat_features
            
        
    # Creating a SHAP DeepExplainer object on the classification head , and providing the sampled training articles as the background dataset
    explainer = shap.DeepExplainer(model.model.classifier, train_features.to(device))
    # Compute the SHAP value for each feature/author pair using the test set
    shap_values = explainer.shap_values(test_features.to(device))
    # Plot a bar chart of the top 5 most important features for each author
    plot_bar(shap_values)
    
# List to map from integer index to label
label_mapping = ['flan_t5_xxl','gpt-3.5-trubo','gpt_j','gpt_neox','human','llama_65B','opt_30b','t0_11b','text-davinci-003']
# Retrieving the parameter of the model with the best hyperparameter configuration
stylo_params = find_best_model()

# Initialising the training and test set
test_set = FakeNewsDataset("preprocessed_data/test.csv", mode="stylo")
train_set = FakeNewsDataset("preprocessed_data/train.csv", mode="stylo")
# Obtaining the name of the columns that contain the stylometric features
stylo_columns = train_set.stylo_features
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = "microsoft/deberta-v3-base"

stylometric_analysis(stylo_params)
