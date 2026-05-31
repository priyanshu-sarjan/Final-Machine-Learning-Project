import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. Load customer dataset using Pandas
# Replace 'customer_data.csv' with your actual dataset filename
df = pd.read_csv('customer_data.csv')

# 2. Handle missing values
# Filling numerical with median and categorical with mode
df['Quantity'] = df['Quantity'].fillna(df['Quantity'].median())
df['UnitPrice'] = df['UnitPrice'].fillna(df['UnitPrice'].median())
df['Country'] = df['Country'].fillna(df['Country'].mode()[0])

# 3. Encode categorical columns using LabelEncoder
le = LabelEncoder()
df['Country'] = le.fit_transform(df['Country'])

# 4. Visualize customer data using Matplotlib
plt.figure(figsize=(10, 6))
sns.histplot(df['Quantity'], bins=30, kde=True, color='skyblue')
plt.title('Customer Distribution by Quantity')
plt.xlabel('Quantity')
plt.ylabel('Frequency')
plt.show()

# 5. Split data into training and testing sets
# Select the specific features requested
X = df[['Quantity', 'UnitPrice', 'Country']]

# CHANGE 'Target' TO YOUR ACTUAL PREDICTION COLUMN NAME
y = df['Target'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling (Crucial for KNN and Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5)
}

accuracies = {}

# 6, 7, 8, 9. Train and Evaluate models
for name, model in models.items():
    # Train
    if name in ["Logistic Regression", "K-Nearest Neighbors"]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    accuracies[name] = acc
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"--- {name} ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Confusion Matrix:\n{cm}\n")

# 10. Compare model performances using graphs
plt.figure(figsize=(8, 5))
sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()), palette='viridis')
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy Score')
plt.ylim(0, 1.0) 
plt.show()
