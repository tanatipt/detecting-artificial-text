from transformers import AutoModelForSequenceClassification
from torch.utils.data import Dataset
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import torch
import pandas as pd
from torch.nn import Linear

        
class DebertaStylo(torch.nn.Module):
    """
    The DebertaStylo class allows us to create the Debertastylo model for neural authorship attribution
    """
    
    def __init__(self, dropout_prob, device):   
        """
        
        Initialises the DebertaStylo model

        Args:
            dropout_prob (float): The dropout probability to be applied to our model
            device (torch.device): The device to place our model on 
        """
        super(DebertaStylo, self).__init__()
        # Defining the vector size to the model's classification head and the number of labels it will have
        out_hidden_size = 825
        num_labels = 9
        
        # Initialising the pre-trained DeBERTa model
        model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v3-base",  hidden_dropout_prob = dropout_prob,attention_probs_dropout_prob= dropout_prob, problem_type = "single_label_classification").to(device)
        # Changing the classification head of the model
        model.classifier = Linear(in_features=out_hidden_size, out_features=num_labels, bias=True).to(device)
        self.model = model
        self.config = model.config
        self.device = device

    def forward(self, X_text ,X_stylo, tokeniser):
        """
        
        Given a batch of news articles, this function performs a forward pass on our model to compute the logit outputs
        for each article. Since there are nine authors, the output dimension will be batch_size*9.

        Args:
            X_text (list): A list of news articles text in the batch
            X_stylo (torch.tensor): A matrix that contains the 57 stylometric features for each news article in the batch
            tokeniser (PreTrainedTokenizer): Tokeniser object for the DeBERTa model to tokenise X_text

        Returns:
            torch.tensor: A matrix that contains the logit outputs for each article in the batch
        """
        # Encoding the text of the news articles into embeddings
        batch_data = create_batch(X_text, tokeniser, self.device, stylo=True)
        # The embeddings of each article is processed through DeBERTa, and we obtain the embedding corresponding to the [CLS] token
        pooler_out = self.model.pooler(self.model.deberta(**batch_data).last_hidden_state)
        X_stylo = X_stylo.to(self.device, dtype = torch.float32)
        # Concatenating the [CLS] embedding and the stylometric features of the article
        concat_input = torch.cat((pooler_out, X_stylo), dim = -1)
        # Passing the concatenated vector through the classification head to obtain the logit outputs
        classifier_out = self.model.dropout(self.model.classifier(concat_input))
    

        return classifier_out

    
class FakeNewsDataset(Dataset):
    """
    
    A Dataset object that

    Args:
        Dataset (_type_): _description_
    """
    def __init__(self, path, mode):
        """
        
        Initialises the FakeNewsDataset object

        Args:
            path (str): A path string leading to the location where the CSV file of the dataset is stored
            mode (str): The mode of the dataset : "ft" or "stylo". The initial value is used when fine-tuning the baseline, whereas
                        the later is used when training the DeBERTa Stylo model.
        """
        self.mode = mode
        # Loading the CSV file from the given location as a dataframe
        self.dataset = pd.read_csv(path)
        # Obtaining the name of the columns that contain the stylometric features
        self.stylo_features = self.dataset.columns[list(self.dataset.columns).index("label")+1:].to_list()
        self.dataset = self.dataset.drop("label", axis=1)
        
        if mode == "ft":
            # When fine-tuning the baseline, we only need the articles text in our feature matrix
            self.X = self.dataset["text"]
            self.y = torch.tensor(self.dataset.drop(columns=["text"] + self.stylo_features).values).float()
        else:
            # When training our model, we need both the articles text and the stylometric features in our feature matrix
            features = ['text'] + self.stylo_features
            self.X = self.dataset[features]
            self.y = torch.tensor(self.dataset.drop(columns=features).values).float()
            
    def __len__(self):
        """
        Returns the number of articles in the dataset

        Returns:
            int: The number of articles in the dataset
        """
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """
        
        Returns the features and label of the article at a given index in the dataset

        Args:
            idx (int): The index at which to retrieve the data from in the dataset

        Returns:
            tuple : if mode = "ft", return article text and its label; otherwise, return article text, its stylometric features and lable
        """
        if self.mode == "ft":
            return self.X.iloc[idx], self.y[idx]
        else:
            row = self.X.iloc[idx]
            text = row["text"]
            stylo_features  = row[self.stylo_features].values.astype(float)
            stylo_tensor = torch.tensor(stylo_features)
            return text, stylo_tensor, self.y[idx]
    
    
def compute_metric(y_pred_class, y_true_class):
    """
    
    Given the predicted and correct authors for a batch of news articles, this function 
    computes the evaluation metrics for the entire batch.

    Args:
        y_pred_class (list): The predicted author for each article in a batch
        y_true_class (list): The correct author for each article in the batch

    Returns:
        tuple: Accuracy, precision, recall and F1 scores
    """
    # Need to specify that there are 9 possible labels, since we have 9 authors in the dataset
    labels = [i for i in range(9)]
    # Computing the accuracy of the batch
    accuracy = accuracy_score(y_true_class, y_pred_class)
    # Computing the macro precision score of the batch
    precision = precision_score(y_true_class, y_pred_class, average="macro", zero_division=0, labels=labels)
    # Computing the macro recall score of the batch
    recall = recall_score(y_true_class, y_pred_class, average="macro",zero_division=0, labels=labels)
    # Computing the macro F1 score of the batch
    f1 = f1_score(y_true_class, y_pred_class, average="macro",zero_division=0, labels=labels)

    return accuracy, precision, recall, f1

def create_batch(X, tokeniser, device,y = None, stylo = False):
    """
    Given a batch of articles text, this function encodes each article as embeddings using the provided tokeniser

    Args:
        X (list): A list of strings to be encoded
        tokeniser (PreTrainedTokenizer) : The tokeniser to be used to encode the strings
        device (torch.device):  The device to place the embeddings on
        y (torch.tensor, optional): A matrix of one-hot encodings for each article in the batch, representing its label . Defaults to None.
        stylo (bool, optional): A flag to indicate whether we are fine-tuning the baselines or training DeBERTa stylo. Defaults to False.

    Returns:
        _type_: _description_
    """
    batch_data = tokeniser(X, padding=True, truncation=True, return_tensors="pt").to(device)
    if not stylo:
        batch_data['labels'] = torch.argmax(y, dim = 1).to(device)
    return batch_data