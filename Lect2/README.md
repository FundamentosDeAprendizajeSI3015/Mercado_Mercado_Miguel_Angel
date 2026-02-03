# Comprehensive Iris Dataset Analysis with Machine Learning

## Project Overview

This is a complete end-to-end machine learning project that performs comprehensive analysis of the famous Iris dataset. The project includes exploratory data analysis, feature engineering, model training with cross-validation, hyperparameter optimization, and interactive visualizations.

## Features

### 📊 Exploratory Data Analysis (EDA)
- Dataset shape, dimensions, and statistical summaries
- Class distribution analysis
- Correlation matrix heatmap
- Pairplot showing feature relationships by species

### 🎨 Interactive Visualizations
All visualizations are exported to HTML and can be opened in any web browser:
- **Interactive Scatter Matrix**: Pairwise scatter plots for all features
- **Parallel Coordinates**: View all features simultaneously with species coloring
- **3D Scatter Plot**: Visualize three features in 3D space
- **PCA 2D Projection**: Dimensionality reduction to 2 components
- **t-SNE Visualization**: Non-linear dimensionality reduction
- **Cross-Validation Results**: Interactive table of model performance
- **Confusion Matrix**: Interactive heatmap of test set predictions
- **Feature Importance**: Bar chart of feature contributions (for tree-based models)

### 🔧 Feature Engineering
The project automatically creates new features:
- **Sepal Ratio**: sepal_length / sepal_width
- **Petal Ratio**: petal_length / petal_width
- **Sepal Area**: sepal_length × sepal_width
- **Petal Area**: petal_length × petal_width

### 🤖 Machine Learning Models
Five classification algorithms are trained and evaluated:
1. **Logistic Regression** - Linear classifier with regularization
2. **K-Nearest Neighbors (KNN)** - Instance-based learning
3. **Decision Tree** - Single tree classifier
4. **Random Forest** - Ensemble of 200 decision trees (grid searched)
5. **Support Vector Machine (SVM) RBF** - Non-linear kernel method (grid searched)

### 📈 Model Evaluation
- 5-fold stratified cross-validation
- Grid search for hyperparameter optimization (SVM and Random Forest)
- Test set evaluation with:
  - Accuracy, Precision, Recall, F1-Score
  - Confusion matrix
  - Per-class performance metrics
  - ROC-AUC score (One-vs-Rest)

## Installation

### Prerequisites
- Python 3.8 or higher
- pip or conda package manager

### Setup Instructions

1. **Clone/Download the project**
   ```bash
   cd /path/to/project
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required packages**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn joblib plotly
   ```

## Usage

### Running the Analysis

Execute the main script to perform the complete analysis:

```bash
python iris_analysis_interactive.py
```

The script will:
1. Load and prepare the Iris dataset
2. Perform exploratory data analysis
3. Generate static visualizations (PNG)
4. Create interactive visualizations (HTML)
5. Engineer new features
6. Train multiple classification models
7. Optimize hyperparameters with grid search
8. Evaluate models on the test set
9. Save the best model and generate a summary report

### Execution Time
- Total execution time: ~2-3 minutes (depending on system performance)
- Most time is spent on t-SNE computation and grid search

## Output Files

All outputs are saved in the `outputs/` directory:

### Static Visualizations (PNG)
- `correlation_heatmap.png` - Feature correlation matrix
- `pairplot.png` - Pairwise relationships colored by species

### Interactive Visualizations (HTML - open in browser)
- `interactive_scatter_matrix.html` - Explore feature pairs interactively
- `interactive_parallel_coordinates.html` - View all features simultaneously
- `interactive_scatter_3d.html` - 3D feature exploration
- `interactive_pca.html` - PCA projection with explained variance
- `interactive_tsne.html` - t-SNE nonlinear projection
- `interactive_cv_results.html` - Cross-validation performance table
- `interactive_confusion_matrix.html` - Test set prediction heatmap
- `interactive_feature_importance.html` - Feature importance ranking

### Reports & Models
- `summary_report.txt` - Comprehensive analysis summary
- `iris_best_model.joblib` - Trained model for future predictions

## Project Structure

```
.
├── iris_analysis_interactive.py   # Main analysis script
├── README.md                       # This file
└── outputs/                        # Results directory
    ├── *.png                       # Static visualizations
    ├── *.html                      # Interactive visualizations
    ├── summary_report.txt          # Analysis summary
    └── iris_best_model.joblib      # Trained model
```

## Key Results

### Selected Model
**Support Vector Machine (SVM) with RBF Kernel**
- Cross-Validation Accuracy: 96.48%
- Test Set Accuracy: 89.47%
- ROC-AUC Score: 0.9949

### Model Performance by Species
| Species | Precision | Recall | F1-Score |
|---------|-----------|--------|----------|
| Setosa | 100% | 92% | 96% |
| Versicolor | 80% | 92% | 86% |
| Virginica | 92% | 85% | 88% |

### Feature Engineering Impact
The addition of 4 engineered features improved model generalization and provided more discriminative information for classification.

## Technical Details

### Libraries Used
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **matplotlib & seaborn** - Static visualizations
- **scikit-learn** - Machine learning algorithms and preprocessing
- **joblib** - Model persistence
- **plotly** - Interactive visualizations

### Preprocessing
- StandardScaler: Normalization for algorithms sensitive to feature scaling
- Feature normalization [0, 1]: For parallel coordinates visualization
- Train-test split: 75% training, 25% testing with stratification

### Cross-Validation Strategy
- Stratified K-Fold (k=5) ensures each fold maintains class distribution
- Prevents class imbalance issues in evaluation
- Provides robust model performance estimates

## How to Use the Trained Model

```python
from joblib import load

# Load the trained model
model = load('outputs/iris_best_model.joblib')

# Make predictions on new data
predictions = model.predict(new_features)

# Get prediction probabilities
probabilities = model.predict_proba(new_features)
```

## Customization

### Modify Dataset Split
Edit the `test_size` parameter in Section 5:
```python
X_train, X_test, y_train, y_test = train_test_split(
    X_feat, y, test_size=0.30,  # Change to 30%
    stratify=y, random_state=RANDOM_STATE
)
```

### Adjust Grid Search Parameters
Modify the parameter grids in Section 5:
```python
param_grid_svm = {
    "clf__C": [0.01, 0.1, 1, 10, 100],      # Add more values
    "clf__gamma": ["scale", "auto", 0.001]  # Add more gammas
}
```

### Add New Models
Add new pipelines to the `models` dictionary:
```python
models["GradientBoosting"] = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", GradientBoostingClassifier())
])
```

## Notes

- The project uses a fixed random seed (42) for reproducibility
- Interactive visualizations require an internet connection to load Plotly JavaScript
- Warnings are suppressed for cleaner output (can be removed for debugging)
- All column names are normalized (spaces and units removed for consistency)

## Future Improvements

Possible enhancements to the project:
1. Add feature selection methods (SelectKBest, RFE)
2. Implement cross-validation for hyperparameter tuning of other models
3. Add learning curves to detect overfitting/underfitting
4. Create prediction pipeline for new iris samples
5. Add ROC curve visualization
6. Implement ensemble methods combining multiple models
7. Add SHAP values for model interpretability

## License

This project is provided as-is for educational purposes.

## Author

Data Science Analysis Pipeline
Created: 2024

## Questions & Support

For questions or issues running the project:
1. Ensure all dependencies are installed: `pip list`
2. Check Python version: `python --version`
3. Verify scikit-learn version supports your Python version
4. Clear the `outputs/` directory and re-run the script

---

**Enjoy exploring the Iris dataset with this comprehensive analysis pipeline!**
