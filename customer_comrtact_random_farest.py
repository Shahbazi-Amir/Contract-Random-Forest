# Import necessary libraries
import pandas as pd

# Define the local path to the dataset
local_path = "bank-additional-full.csv"  # Make sure the file is in the same directory as your notebook

# Load the dataset from the local file
try:
    data = pd.read_csv(local_path, sep=';')
    print("Dataset loaded successfully!")
except Exception as e:
    print(f"Error loading dataset: {e}")
    
# Check for missing values
missing_values = data.isnull().sum()

# Display information about the dataset
data_info = data.info()

# Display summary statistics
summary_stats = data.describe(include='all')

missing_values, data_info, summary_stats



# Import necessary libraries
import pandas as pd

# Apply One-Hot Encoding to categorical columns
categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']
data_encoded = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

# Display the first few rows of the encoded dataset
data_encoded.head()


# Check the distribution of the target column
target_distribution = data['y'].value_counts()
print("Target Distribution:\n", target_distribution)

# Apply SMOTE to balance the dataset
from imblearn.over_sampling import SMOTE

# Separate features and target
X = data_encoded.drop(columns=['y'])
y = data_encoded['y']

# Apply SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Check the new distribution after SMOTE
resampled_distribution = pd.Series(y_resampled).value_counts()
print("\nResampled Distribution:\n", resampled_distribution)



# Import necessary libraries
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Display the shapes of the datasets
print("Training set shape:", X_train.shape, y_train.shape)
print("Testing set shape:", X_test.shape, y_test.shape)


# Check the distribution of classes in the training and testing sets
print("Training set class distribution:\n", y_train.value_counts())
print("\nTesting set class distribution:\n", y_test.value_counts())



# Import necessary libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Train the Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)



# Get feature importances from the trained model
import pandas as pd
import matplotlib.pyplot as plt

# Extract feature importances
feature_importances = model.feature_importances_

# Create a DataFrame for visualization
feature_importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# Display the top 10 most important features
print("Top 10 Most Important Features:")
print(feature_importance_df.head(10))

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'][:10], feature_importance_df['Importance'][:10], color='skyblue')
plt.gca().invert_yaxis()  # Invert y-axis for better readability
plt.title('Top 10 Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()


# Evaluate the model on the test set
y_pred = model.predict(X_test)

from sklearn.metrics import classification_report
print("\nClassification Report on Test Set:")
print(classification_report(y_test, y_pred))