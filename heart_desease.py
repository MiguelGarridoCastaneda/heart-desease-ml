#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %%
df = pd.read_csv("heart.csv")
df.head()
# %%
df.info()
# %%
sns.heatmap(df.corr())

# %%
from sklearn.model_selection import train_test_split

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# %%
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()

# %%
clf.fit(X_train, y_train)
clf.score(X_test, y_test)

# %%
predictions = clf.predict(X_test)


# %%
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, f1_score

print("Confusion Matrix: ")
print(confusion_matrix(y_test, predictions))
print(f"Accuracy value: {accuracy_score(y_test, predictions)}")
print(f"Recall value: {recall_score(y_test, predictions)}")
print(f"F1_score value: {f1_score(y_test, predictions)}")
# %%
