# USAGE EXAMPLES - Iris Analysis Pipeline

This file contains practical examples of how to use the trained model and explore the analysis results.

## Example 1: Making Predictions with the Trained Model

```python
import numpy as np
from joblib import load
import pandas as pd

# Load the trained SVM model
model = load('outputs/iris_best_model.joblib')

# Example 1a: Predict a single iris flower
# Feature order: sepal_length, sepal_width, petal_length, petal_width,
#                sepal_ratio, petal_ratio, sepal_area, petal_area

new_iris = np.array([[5.1, 3.5, 1.4, 0.2, 1.4571, 7.0, 17.85, 0.28]])
prediction = model.predict(new_iris)
probability = model.predict_proba(new_iris)[0]

print(f"Predicted species: {prediction[0]}")
print(f"Prediction confidence: {probability.max():.2%}")
print(f"Class probabilities: setosa={probability[0]:.2%}, versicolor={probability[1]:.2%}, virginica={probability[2]:.2%}")
```

## Example 2: Batch Predictions

```python
import pandas as pd
from joblib import load

# Load model
model = load('outputs/iris_best_model.joblib')

# Create DataFrame with new samples (including engineered features)
new_samples = pd.DataFrame({
    'sepal_length': [5.1, 6.2, 7.1],
    'sepal_width': [3.5, 2.9, 3.0],
    'petal_length': [1.4, 4.3, 5.9],
    'petal_width': [0.2, 1.3, 2.1],
    'sepal_ratio': [1.4571, 2.1379, 2.3667],
    'petal_ratio': [7.0, 3.3077, 2.8095],
    'sepal_area': [17.85, 17.98, 21.3],
    'petal_area': [0.28, 5.59, 12.39]
})

predictions = model.predict(new_samples)
probabilities = model.predict_proba(new_samples)

# Display results
results = pd.DataFrame({
    'Prediction': predictions,
    'Confidence': probabilities.max(axis=1),
    'Setosa_Prob': probabilities[:, 0],
    'Versicolor_Prob': probabilities[:, 1],
    'Virginica_Prob': probabilities[:, 2]
})

print(results)
```

## Example 3: Viewing Cross-Validation Results

```python
import pandas as pd

# Read the summary report
with open('outputs/summary_report.txt', 'r') as f:
    report = f.read()
    print(report)

# Or extract specific sections
cv_section = report.split('SELECTED MODEL')[0]
print("Cross-Validation Results:")
print(cv_section)
```

## Example 4: Loading and Exploring the Analysis

```python
import plotly.io as pio

# Open interactive visualizations in your default browser
import webbrowser

# List of available interactive plots
plots = {
    'Scatter Matrix': 'outputs/interactive_scatter_matrix.html',
    'Parallel Coordinates': 'outputs/interactive_parallel_coordinates.html',
    '3D Scatter': 'outputs/interactive_scatter_3d.html',
    'PCA 2D': 'outputs/interactive_pca.html',
    't-SNE': 'outputs/interactive_tsne.html',
    'Confusion Matrix': 'outputs/interactive_confusion_matrix.html',
    'CV Results': 'outputs/interactive_cv_results.html'
}

# Open the scatter matrix in your browser
webbrowser.open(plots['Scatter Matrix'])
```

## Example 5: Re-training with Different Parameters

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn import datasets
from joblib import dump

# Load and prepare data
iris = datasets.load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)

# Add engineered features
X.columns = [c.replace(" (cm)", "").replace(" ", "_") for c in X.columns]
X["sepal_ratio"] = X["sepal_length"] / X["sepal_width"]
X["petal_ratio"] = X["petal_length"] / X["petal_width"]
X["sepal_area"] = X["sepal_length"] * X["sepal_width"]
X["petal_area"] = X["petal_length"] * X["petal_width"]

y = iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

# Define SVM pipeline
svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', SVC(kernel='rbf', probability=True, random_state=42))
])

# New grid search with different parameters
param_grid = {
    'clf__C': [0.01, 0.1, 1, 10, 100, 1000],
    'clf__gamma': ['scale', 'auto', 0.0001, 0.001, 0.01, 0.1]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(svm_pipeline, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")
print(f"Test score: {grid_search.score(X_test, y_test):.4f}")

# Save new model
dump(grid_search.best_estimator_, 'outputs/iris_best_model_new.joblib')
```

## Example 6: Feature Engineering Analysis

```python
import pandas as pd
from sklearn import datasets
import numpy as np

# Load data
iris = datasets.load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)

# Clean column names
X.columns = [c.replace(" (cm)", "").replace(" ", "_") for c in X.columns]

# Create engineered features
X_eng = X.copy()
X_eng["sepal_ratio"] = X["sepal_length"] / X["sepal_width"]
X_eng["petal_ratio"] = X["petal_length"] / X["petal_width"]
X_eng["sepal_area"] = X["sepal_length"] * X["sepal_width"]
X_eng["petal_area"] = X["petal_length"] * X["petal_width"]

# Compare statistics
print("Original Features Statistics:")
print(X.describe())

print("\nEngineered Features Statistics:")
print(X_eng[["sepal_ratio", "petal_ratio", "sepal_area", "petal_area"]].describe())

# Correlations with target
y = datasets.load_iris().target
correlations = X_eng.corrwith(pd.Series(y))
print("\nFeature-Target Correlations:")
print(correlations.sort_values(ascending=False))
```

## Example 7: Model Comparison

```python
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Load and prepare data
iris = datasets.load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
X.columns = [c.replace(" (cm)", "").replace(" ", "_") for c in X.columns]
X["sepal_ratio"] = X["sepal_length"] / X["sepal_width"]
X["petal_ratio"] = X["petal_length"] / X["petal_width"]
X["sepal_area"] = X["sepal_length"] * X["sepal_width"]
X["petal_area"] = X["petal_length"] * X["petal_width"]

y = iris.target

# Define models
models = {
    'Logistic Regression': Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression())]),
    'KNN': Pipeline([('scaler', StandardScaler()), ('clf', KNeighborsClassifier())]),
    'Decision Tree': Pipeline([('clf', DecisionTreeClassifier())]),
    'Random Forest': Pipeline([('clf', RandomForestClassifier(n_estimators=200))]),
    'SVM RBF': Pipeline([('scaler', StandardScaler()), ('clf', SVC(kernel='rbf'))])
}

# Cross-validate all models
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    results[name] = {
        'Mean': scores.mean(),
        'Std': scores.std(),
        'Scores': scores
    }

# Display results
df_results = pd.DataFrame({
    'Model': results.keys(),
    'Mean Accuracy': [results[m]['Mean'] for m in results.keys()],
    'Std Dev': [results[m]['Std'] for m in results.keys()]
})

print(df_results.sort_values('Mean Accuracy', ascending=False))
```

## Example 8: Visualization with Matplotlib

```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn import datasets

# Load data
iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target_names[iris.target]

# Rename columns
df.columns = [c.replace(" (cm)", "").replace(" ", "_") for c in df.columns]

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Scatter plot 1
axes[0, 0].scatter(df['sepal_length'], df['sepal_width'], c=iris.target, cmap='viridis')
axes[0, 0].set_xlabel('Sepal Length')
axes[0, 0].set_ylabel('Sepal Width')
axes[0, 0].set_title('Sepal Measurements')

# Scatter plot 2
axes[0, 1].scatter(df['petal_length'], df['petal_width'], c=iris.target, cmap='viridis')
axes[0, 1].set_xlabel('Petal Length')
axes[0, 1].set_ylabel('Petal Width')
axes[0, 1].set_title('Petal Measurements')

# Histograms
df['sepal_length'].hist(ax=axes[1, 0], bins=20)
axes[1, 0].set_title('Sepal Length Distribution')

df['petal_length'].hist(ax=axes[1, 1], bins=20)
axes[1, 1].set_title('Petal Length Distribution')

plt.tight_layout()
plt.show()
```

## Example 9: Extract Confusion Matrix Data

```python
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from joblib import load

# Load model and data
model = load('outputs/iris_best_model.joblib')
iris = datasets.load_iris()

# Prepare data with engineered features
X = pd.DataFrame(iris.data, columns=iris.feature_names)
X.columns = [c.replace(" (cm)", "").replace(" ", "_") for c in X.columns]
X["sepal_ratio"] = X["sepal_length"] / X["sepal_width"]
X["petal_ratio"] = X["petal_length"] / X["petal_width"]
X["sepal_area"] = X["sepal_length"] * X["sepal_width"]
X["petal_area"] = X["petal_length"] * X["petal_width"]

y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

# Get predictions and confusion matrix
from sklearn.metrics import confusion_matrix
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

# Display as DataFrame
cm_df = pd.DataFrame(cm, 
                     index=[f'True {iris.target_names[i]}' for i in range(3)],
                     columns=[f'Pred {iris.target_names[i]}' for i in range(3)])

print(cm_df)
```

## Example 10: Understanding Model Predictions

```python
import numpy as np
from joblib import load
import pandas as pd

# Load model
model = load('outputs/iris_best_model.joblib')

# Create a sample iris flower
sample = pd.DataFrame({
    'sepal_length': [7.0],
    'sepal_width': [3.2],
    'petal_length': [4.7],
    'petal_width': [1.4],
    'sepal_ratio': [7.0 / 3.2],
    'petal_ratio': [4.7 / 1.4],
    'sepal_area': [7.0 * 3.2],
    'petal_area': [4.7 * 1.4]
})

# Make prediction
prediction = model.predict(sample)[0]
probabilities = model.predict_proba(sample)[0]

# Get decision function (for SVM)
if hasattr(model.named_steps['clf'], 'decision_function'):
    decision = model.named_steps['clf'].decision_function(sample)[0]
    print(f"Decision function values: {decision}")

# Display results
species_names = ['setosa', 'versicolor', 'virginica']
print(f"\nPredicted species: {species_names[prediction]}")
print(f"\nProbabilities:")
for species, prob in zip(species_names, probabilities):
    print(f"  {species}: {prob:.4f}")
    
# Confidence level
max_prob_idx = np.argmax(probabilities)
second_prob = probabilities[probabilities != probabilities.max()].max()
confidence = probabilities[max_prob_idx] - second_prob
print(f"\nConfidence (margin): {confidence:.4f}")
```

---

These examples demonstrate common use cases for the trained model and analysis results.
For more information, see the README.md file.
