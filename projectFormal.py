import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the CSV file
csv_file_path = '/home/rev/Downloads/emotions.csv'  # Update with the correct path to your file

data = pd.read_csv(csv_file_path)

# Explore the dataset
print(data.head())
print(data.info())
print(data.describe())
print(data['label'].value_counts())

# Convert labels to numerical values
label_encoder = LabelEncoder()
data['label'] = label_encoder.fit_transform(data['label'])

# Split the data into features and labelsp
X = data.drop('label', axis=1)
y = data['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Develop the Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_report_text = classification_report(y_test, y_pred)
confusion_matrix_values = confusion_matrix(y_test, y_pred)

# Print results
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report_text)
print("Confusion Matrix:\n", confusion_matrix_values)

# Visualize the Confusion Matrix
plt.figure(figsize=(10, 7))
sns.heatmap(confusion_matrix_values, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()  