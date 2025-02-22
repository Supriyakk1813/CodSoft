#TASK ONE TITANIC DATASET
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

# Select relevant features
a = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
b = df['Survived']

# Handle missing values
a.loc[:, 'Age'] = a['Age'].fillna(a['Age'].mean())

# Convert categorical data
a['Sex'] = LabelEncoder().fit_transform(X['Sex'])

# Split dataset
a_train, a_test, b_train, b_test = train_test_split(a, b, test_size=0.2, random_state=42)

# Train model
model = GradientBoostingClassifier()
model.fit(a_train, b_train)

# Predict on test set
b_pred = model.predict(a_test)

# Calculate accuracy
accuracy = accuracy_score(b_test, b_pred)
print(f'Model Accuracy: {accuracy:.2f}')

# Plot results
labels = ['Correct Predictions', 'Incorrect Predictions']
counts = [(b_test == b_pred).sum(), (b_test != b_pred).sum()]

plt.bar(labels, counts, color=['green', 'red'])
plt.alabel('Prediction Type')
plt.blabel('Count')
plt.title('Model Prediction Accuracy')
plt.show()

# Make a single prediction
test_sample = np.array([[3, 1, 25, 0, 0, 7.25]])  # Example: 3rd class, Male, 25 years old, no family, low fare
prediction = model.predict(test_sample)[0]

print("Single Prediction:", "Survived" if prediction == 1 else "Not Survived")