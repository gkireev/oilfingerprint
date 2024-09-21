import os
import pandas as pd
import numpy as np
from models.data_preprocessing import preprocessing
import matplotlib.pyplot as plt
from scipy.stats import f_oneway
import seaborn as sns

path_to_data = os.path.join(os.getcwd(), "biomarker_dataset.xlsx")
data = pd.read_excel(path_to_data)

X, y, freq = preprocessing(data, dist=True)

# Oil group distribution
fig_1, ax_1 = plt.subplots()
ax_1.pie(freq.values, labels=freq.index, autopct = '%1.1f%%')
ax_1.set_title("Oil group distribution")

# One-Way ANOVA ("Depth" and "Oil Group")
df = pd.DataFrame({"Depth": X[:,0], "Oil Group": y})
dic = df.groupby("Oil Group")["Depth"].apply(list).to_dict()
one_way = []
for values in dic.values():
    one_way.append(values[:np.min(freq.values)+1])
print(f_oneway(*one_way))

# Boxplots ("Depth" and "Oil Group")
fig_2, ax_2 = plt.subplots()
sns.boxplot(x="Oil Group", y="Depth", data=df, palette="bright")
plt.show()