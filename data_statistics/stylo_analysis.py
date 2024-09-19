import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Loading the preprocessed train, validation and test dataset
train_data = pd.read_csv("../preprocessed_data/train.csv")
valid_data = pd.read_csv("../preprocessed_data/valid.csv")
test_data = pd.read_csv("../preprocessed_data/test.csv")

# Concatenating the train, validation and test set to from a single dataset
data = pd.concat([train_data, valid_data, test_data])
# Obtaining the name of the columns that contain the stylometric features
stylo_features = data.columns[list(data.columns).index("label")+1:]
# Obtaining only the stylometric features of the articles from the concatenated dataset
stylo_matrix = data[stylo_features].to_numpy()
tsne = TSNE(n_components=2)
# Fitting the TSNE with 2 components to the stylometric features of all articles
X = tsne.fit_transform(stylo_matrix)

# For each author, we plot the projected stylometric features of all articles they have generated.
for label in np.unique(data['label']):
    label_idx = data.index[data['label'] == label].to_list()
    X_label = X[label_idx]
    plt.scatter(X_label[:, 0], X_label[:, 1], label=label, marker="." , s=25)

plt.legend(fontsize=12)
plt.title("TSNE Stylometric Features")
plt.ylabel("TSNE Component 2", fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("TSNE Component 1", fontsize=12)
plt.xticks(fontsize=12)
plt.tight_layout()
plt.show()
#plt.savefig("results/stylometry_plot.png")
plt.close()