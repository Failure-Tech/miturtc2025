import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

# loading data
df = pd.read_csv("../GSE33000_raw_data.txt", sep="\t", comment="!", index_col=0)

df = df.dropna(how="all")

df_expr = df.T

# removing low-variance genes
gene_variance = df.iloc[:, 3:].var(axis=1)

threshold = gene_variance > 0.01
filtered_df = df[threshold]

# print(filtered_df.head())

# z-score normalization
metadata = filtered_df.iloc[:, :3]
numerical_data = filtered_df.iloc[:, 3:]
numerical_data_T = numerical_data.T

scaler = StandardScaler()

z_scores = scaler.fit_transform(numerical_data_T)

normalized_df = pd.DataFrame(z_scores, columns=numerical_data.index[:z_scores.shape[-1]])
normalized_df = normalized_df.T
normalized_df.columns = numerical_data.columns
final_df = pd.concat([metadata, normalized_df], axis=1)
print(final_df.head())

# label encoder
n_samples = final_df.shape[0]
print("N_samples", n_samples)
# labels = np.array([0]*int(n_samples/2) + [1]*int(n_samples/2))
labels = np.zeros(n_samples, dtype=int)
labels[n_samples//2 :] = 1

final_df["label"] = labels

X = final_df.iloc[:, 3:-1].values
y = final_df["label"].values

print(f"{X} \n {y}")

X_df = pd.DataFrame(X)
X_df["label"] = y

X_df.to_csv("../processsed.csv", index=False)