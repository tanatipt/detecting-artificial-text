from transformers import AutoTokenizer, AutoModel
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np 
import torch

checkpoint = "microsoft/deberta-v3-base"
# Loading the preprocessed train, validation and test dataset
train_data = pd.read_csv("../preprocessed_data/train.csv")
valid_data = pd.read_csv("../preprocessed_data/valid.csv")
test_data = pd.read_csv("../preprocessed_data/test.csv")
# Concatenating the train, validation and test set to from a single dataset
data = pd.concat([train_data, valid_data, test_data])
# Obtaining only the text of the articles from the concatenated dataset
corpus = data['text'].to_list()

# Initialsing the pre-trained DeBERTa model and its tokeniser
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokeniser = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModel.from_pretrained(checkpoint).to(device)

# A matrix to store the [CLS] embedding of each article in the concatenated dataset
cls_matrix = []
with torch.no_grad():
    
    # Iterating through the list of articles text
    for idx, text in enumerate(corpus):
        print("Index : ", idx)
        # Encoding a the text of the article into embeddings
        batch_data = tokeniser(text, padding=True, truncation=True, return_tensors="pt").to(device)
        # Passing the embeddings to the DeBERTa model to generate the [CLS] embedding of the article
        cls_token = model(**batch_data).last_hidden_state[:, 0, :].flatten().tolist()
        # Storing the [CLS] embedding of the article
        cls_matrix.append(cls_token)
        
cls_matrix = np.array(cls_matrix)      
tsne = TSNE(n_components=2)
# Fitting the TSNE with 2 components to the [CLS] embedding of all articles
X = tsne.fit_transform(cls_matrix)

# For each author, we plot the projected [CLS] embeddings of all articles they have generated.
for label in np.unique(data['label']):
    label_idx = data.index[data['label'] == label].to_list()
    X_label = X[label_idx]
    plt.scatter(X_label[:, 0], X_label[:, 1], label=label, marker="." , s=25)

plt.legend(fontsize=12)
plt.title("TSNE Embedding Space of DeBERTa")
plt.ylabel("TSNE Component 2", fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("TSNE Component 1", fontsize=12)
plt.xticks(fontsize=12)
plt.tight_layout()
plt.show()
#plt.savefig("results/cls_embedding_plot.png")

plt.close()