import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter

# Load data
data_dict = pickle.load(open('./data.pickle', 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Check class distribution
unique, counts = np.unique(labels, return_counts=True)
print("Class distribution before filtering:", dict(zip(unique, counts)))

# Remove classes with less than 2 samples
label_counts = Counter(labels)
valid_classes = {key for key, value in label_counts.items() if value > 1}
filtered_indices = [i for i in range(len(labels)) if labels[i] in valid_classes]

# Apply filtering
data = data[filtered_indices]
labels = labels[filtered_indices]

# Print updated class distribution
unique, counts = np.unique(labels, return_counts=True)
print("Class distribution after filtering:", dict(zip(unique, counts)))

# Split data (ensure stratify is valid)
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels if len(set(labels)) > 1 else None
)

# Train model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Predict and evaluate
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)

print(f'{score * 100:.2f}% of samples were classified correctly!')

# Save model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)