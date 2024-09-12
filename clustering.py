import os
import pandas as pd
import numpy as np
from data_preprocessing import preprocessing
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

path_to_data = os.path.join(os.getcwd(), "biomarker_dataset.xlsx")
data = pd.read_excel(path_to_data)

X, y = preprocessing(data)

# t-SNE
tsne = TSNE(n_components=2, random_state=1)
X_tsne = tsne.fit_transform(X)

# K-means (n_clusters = the number of unique oil groups)
kmeans = KMeans(n_clusters=len(np.unique(y)), random_state=0)
kmeans.fit(X_tsne)

# Results visualisation
X_df = pd.DataFrame(data=X_tsne, columns=["Dim1", "Dim2"])
X_df["Label"] = kmeans.labels_

sns.scatterplot(x="Dim1", y="Dim2", data=X_df, hue="Label", palette="bright")
plt.title("Oil clustering")

target = list(y)
for i in range(len(target)):
    plt.text(X_tsne[i, 0], X_tsne[i, 1], target[i][-2:])

plt.show()
