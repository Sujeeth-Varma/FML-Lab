import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

# Load dataset
msg = pd.read_csv("./datasets/5.csv", names=['message', 'label'])
print("Total Instances of Dataset:", msg.shape[0])

# Ensure no whitespace issues in labels
msg['label'] = msg['label'].str.strip()

# Convert labels to numerical values
msg['labelnum'] = msg['label'].map({'pos': 1, 'neg': 0})

# Check for NaN values and drop them
print("Rows with NaN labelnum:", msg[msg['labelnum'].isna()])
msg = msg.dropna(subset=['labelnum'])
msg['labelnum'] = msg['labelnum'].astype(int)

# Split dataset
X = msg['message']
y = msg['labelnum']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature extraction
count_vectorizer = CountVectorizer()
X_train_dm = count_vectorizer.fit_transform(X_train)
X_test_dm = count_vectorizer.transform(X_test)

# Convert to DataFrame for better visualization
df = pd.DataFrame(X_train_dm.toarray(), columns=count_vectorizer.get_feature_names_out())
print(df.head())

# Train Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train_dm, y_train)

# Make predictions
pred = clf.predict(X_test_dm)

# Debugging output for y_test and predictions
print("y_test contains NaN values:", y_test.isna().sum())
print("Unique y_test values:", y_test.unique())

# Display predictions
for doc, p in zip(X_test, pred):
    label = 'pos' if p == 1 else 'neg'
    print(f"{doc} -> {label}")

# Evaluate model
print('\nAccuracy Metrics:')
print('Accuracy:', accuracy_score(y_test, pred))
print('Recall:', recall_score(y_test, pred))
# print('Precision:', precision_score(y_test, pred))
print('Precision:', precision_score(y_test, pred, zero_division=1))
print('Confusion Matrix:\n', confusion_matrix(y_test, pred))
