import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
df = pd.read_csv(r"./datasets/9.csv" )
df.head()
X = df.drop("species", axis=1)
X.head()
y = df["species"]
y.head()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
y_pred
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)
classifier2 = KNeighborsClassifier(n_neighbors=50)
classifier2.fit(X_train, y_train)
y_pred2 = classifier2.predict(X_test)
y_pred2
acc2 = accuracy_score(y_test, y_pred2)
print("Accuracy:", acc2)
cm = confusion_matrix(y_test, y_pred)
print(cm)
sns.heatmap(cm, annot=True)
cm2 = confusion_matrix(y_test, y_pred2)
print(cm2)
sns.heatmap(cm2, annot=True)