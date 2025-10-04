# ============================================== MULTINOMIAL LOGISTIC REGRESSION - ENHANCED ==============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           classification_report, confusion_matrix, precision_recall_fscore_support)
import mlflow
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv("data/winequality-red.csv", sep=";")

# Create quality groups
def simplify_quality(q):
    if q <= 4:  # low
        return 0
    if q <= 6:  # medium
        return 1 
    else:  # high
        return 2

df["qualityg"] = df["quality"].apply(simplify_quality)

# Prepare data
X = df.drop(columns=["quality", "qualityg"])
y = df["qualityg"]

print("=== DATA PREPARATION ===")
print(f"Dataset shape: {df.shape}")
print(f"Features: {X.shape[1]}")
print(f"Target distribution: {np.bincount(y)}")
print(f"Target classes: Low (0), Medium (1), High (2)")

# Split data with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print(f"Target distribution in training: {np.bincount(y_train)}")
print(f"Target distribution in test: {np.bincount(y_test)}")

# ============================================== MULTINOMIAL REGRESSION WITH PREPROCESSING ==============================================

print("\n" + "="*80)
print("MULTINOMIAL REGRESSION WITH PREPROCESSING")
print("="*80)

# Create pipeline with scaling and multinomial regression
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('multinomial_lr', LogisticRegression(solver='lbfgs', random_state=42))
])

# Fit the pipeline
pipeline.fit(X_train, y_train)

# Make predictions
y_pred_pipeline = pipeline.predict(X_test)
y_pred_proba = pipeline.predict_proba(X_test)

# Calculate metrics
accuracy_pipeline = accuracy_score(y_test, y_pred_pipeline)
precision_pipeline = precision_score(y_test, y_pred_pipeline, average='weighted', zero_division=0)
recall_pipeline = recall_score(y_test, y_pred_pipeline, average='weighted', zero_division=0)
f1_pipeline = f1_score(y_test, y_pred_pipeline, average='weighted', zero_division=0)

print("=== MULTINOMIAL LOGISTIC REGRESSION WITH PREPROCESSING ===")
print(f"Accuracy: {accuracy_pipeline:.4f}")
print(f"Precision: {precision_pipeline:.4f}")
print(f"Recall: {recall_pipeline:.4f}")
print(f"F1 Score: {f1_pipeline:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_pipeline, target_names=['Low', 'Medium', 'High']))

# ============================================== HYPERPARAMETER TUNING FOR MULTINOMIAL REGRESSION ==============================================

print("\n" + "="*80)
print("HYPERPARAMETER TUNING FOR MULTINOMIAL REGRESSION")
print("="*80)

# Define parameter grid for hyperparameter tuning
param_grid = {
    'multinomial_lr__C': [0.1, 1, 10, 100],
    'multinomial_lr__max_iter': [500, 1000, 2000],
    'multinomial_lr__solver': ['lbfgs', 'newton-cg']
}

# Create GridSearchCV object
grid_search = GridSearchCV(
    pipeline, 
    param_grid, 
    cv=5, 
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=1
)

# Fit the grid search
print("Performing hyperparameter tuning...")
grid_search.fit(X_train, y_train)

# Get best parameters and score
print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Use best model for predictions

best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)
# Calculate metrics for best model
accuracy_best = accuracy_score(y_test, y_pred_best)
precision_best = precision_score(y_test, y_pred_best, average='weighted', zero_division=0)
recall_best = recall_score(y_test, y_pred_best, average='weighted', zero_division=0)
f1_best = f1_score(y_test, y_pred_best, average='weighted', zero_division=0)

print("\n=== BEST MULTINOMIAL LOGISTIC REGRESSION MODEL ===")
print(f"Accuracy: {accuracy_best:.4f}")
print(f"Precision: {precision_best:.4f}")
print(f"Recall: {recall_best:.4f}")
print(f"F1 Score: {f1_best:.4f}")

# ============================================== CROSS-VALIDATION FOR MULTINOMIAL REGRESSION ==============================================

print("\n" + "="*80)
print("CROSS-VALIDATION FOR MULTINOMIAL REGRESSION")
print("="*80)

# Perform cross-validation on the best model
cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='f1_weighted')

print("=== CROSS-VALIDATION RESULTS ===")
print(f"CV F1 Scores: {cv_scores}")
print(f"Mean CV F1 Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Compare with different solvers
solvers = ['lbfgs', 'newton-cg', 'sag', 'saga']
solver_scores = {}

print("\n=== SOLVER COMPARISON ===")
for solver in solvers:
    try:
        model_solver = Pipeline([
            ('scaler', StandardScaler()),
            ('multinomial_lr', LogisticRegression(solver=solver, max_iter=1000, random_state=42))
        ])
        scores = cross_val_score(model_solver, X, y, cv=5, scoring='f1_weighted')
        solver_scores[solver] = scores.mean()
        print(f"{solver}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    except Exception as e:
        print(f"{solver}: Error - {e}")

# Find best solver
if solver_scores:
    best_solver = max(solver_scores, key=solver_scores.get)
    print(f"\nBest solver: {best_solver} with score: {solver_scores[best_solver]:.4f}")

# ============================================== FEATURE IMPORTANCE FOR MULTINOMIAL REGRESSION ==============================================

print("\n" + "="*80)
print("FEATURE IMPORTANCE FOR MULTINOMIAL REGRESSION")
print("="*80)

# Get feature importance from the best model
feature_names = X.columns
coefficients = best_model.named_steps['multinomial_lr'].coef_

# Calculate feature importance for each class
class_names = ['Low', 'Medium', 'High']
feature_importance_df = pd.DataFrame(coefficients.T, index=feature_names, columns=class_names)

print("=== FEATURE IMPORTANCE BY CLASS ===")
print(feature_importance_df)

# Plot feature importance for each class
plt.figure(figsize=(12, 8))
feature_importance_df.plot(kind='bar', figsize=(12, 6))
plt.title('Feature Importance by Class - Multinomial Logistic Regression')
plt.xlabel('Features')
plt.ylabel('Coefficient Value')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Wine Quality Class')
plt.tight_layout()
plt.show()

# Overall feature importance (average absolute coefficient across classes)
overall_importance = np.abs(coefficients).mean(axis=0)
feature_importance_overall = pd.DataFrame({
    'Feature': feature_names,
    'Importance': overall_importance
}).sort_values('Importance', ascending=False)

print("\n=== OVERALL FEATURE IMPORTANCE ===")
print(feature_importance_overall)

# Plot overall feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_overall['Feature'], feature_importance_overall['Importance'])
plt.xlabel('Average Absolute Coefficient')
plt.title('Overall Feature Importance - Multinomial Logistic Regression')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# ============================================== CONFUSION MATRIX AND CLASSIFICATION ANALYSIS ==============================================

print("\n" + "="*80)
print("CONFUSION MATRIX AND CLASSIFICATION ANALYSIS")
print("="*80)

# Create confusion matrix for the best model
cm_best = confusion_matrix(y_test, y_pred_best)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm_best, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Low', 'Medium', 'High'],
            yticklabels=['Low', 'Medium', 'High'])
plt.title('Confusion Matrix - Best Multinomial Logistic Regression Model')
plt.xlabel('Predicted Quality')
plt.ylabel('Actual Quality')
plt.show()

# Class-wise performance analysis
print("=== CLASS-WISE PERFORMANCE ANALYSIS ===")
precision_arr, recall_arr, f1_arr, support_arr = precision_recall_fscore_support(y_test, y_pred_best, labels=[0, 1, 2], zero_division=0)
    
for i, class_name in enumerate(['Low', 'Medium', 'High']):    
    print(f"\n{class_name} Quality Wines:")
    print(f"  Precision: {precision_arr[i]:.4f}")
    print(f"  Recall: {recall_arr[i]:.4f}")
    print(f"  F1-Score: {f1_arr[i]:.4f}")
    print(f"  Support: {support_arr[i]} samples")

# Prediction probabilities analysis
y_pred_proba_best = best_model.predict_proba(X_test)
print(f"\n=== PREDICTION PROBABILITIES ANALYSIS ===")
print(f"Average confidence for correct predictions: {np.mean(np.max(y_pred_proba_best, axis=1)):.4f}")
if np.sum(y_test != y_pred_best) > 0:
    print(f"Average confidence for incorrect predictions: {np.mean(np.max(y_pred_proba_best[y_test != y_pred_best], axis=1)):.4f}")

# ============================================== MLFLOW LOGGING FOR ENHANCED MULTINOMIAL REGRESSION ==============================================

print("\n" + "="*80)
print("MLFLOW LOGGING FOR ENHANCED MULTINOMIAL REGRESSION")
print("="*80)

# Import additional MLflow dependencies
from mlflow.models.signature import infer_signature

# Log the enhanced multinomial regression model with MLflow
with mlflow.start_run(run_name="Enhanced_Multinomial_Regression"):
    # Log parameters
    mlflow.log_param("model_type", "multinomial_logistic_regression")
    mlflow.log_param("solver", best_model.named_steps['multinomial_lr'].solver)
    mlflow.log_param("max_iter", best_model.named_steps['multinomial_lr'].max_iter)
    mlflow.log_param("C", best_model.named_steps['multinomial_lr'].C)
    mlflow.log_param("preprocessing", "StandardScaler")
    mlflow.log_param("multi_class", "auto_multinomial")
    mlflow.log_param("random_state", 42)
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("cv_folds", 5)
    
    # Log hyperparameter tuning parameters
    mlflow.log_param("tuning_method", "GridSearchCV")
    mlflow.log_param("param_grid_C", str([0.1, 1, 10, 100]))
    mlflow.log_param("param_grid_max_iter", str([500, 1000, 2000]))
    mlflow.log_param("param_grid_solver", str(['lbfgs', 'newton-cg']))
    
    # Log metrics
    mlflow.log_metric("accuracy", accuracy_best)
    mlflow.log_metric("precision_weighted", precision_best)
    mlflow.log_metric("recall_weighted", recall_best)
    mlflow.log_metric("f1_weighted", f1_best)
    mlflow.log_metric("cv_f1_mean", cv_scores.mean())
    mlflow.log_metric("cv_f1_std", cv_scores.std())
    mlflow.log_metric("best_cv_score", grid_search.best_score_)
    
    # Log class-wise metrics
    class_precision = precision_score(y_test, y_pred_best, average=None, zero_division=0)
    class_recall = recall_score(y_test, y_pred_best, average=None, zero_division=0)
    class_f1 = f1_score(y_test, y_pred_best, average=None, zero_division=0)
    
    for i, class_name in enumerate(['Low', 'Medium', 'High']):
        mlflow.log_metric(f"precision_{class_name.lower()}", class_precision[i])
        mlflow.log_metric(f"recall_{class_name.lower()}", class_recall[i])
        mlflow.log_metric(f"f1_{class_name.lower()}", class_f1[i])
    
    # Log feature importance
    feature_importance_dict = dict(zip(feature_names, overall_importance))
    mlflow.log_dict(feature_importance_dict, "feature_importance.json")
    
    # Log confusion matrix as artifact
    cm_df = pd.DataFrame(cm_best, 
                        index=['Actual_Low', 'Actual_Medium', 'Actual_High'],
                        columns=['Pred_Low', 'Pred_Medium', 'Pred_High'])
    mlflow.log_dict(cm_df.to_dict(), "confusion_matrix.json")
    
    # Prepare data for signature and input example
    X_test_float = X_test.astype({col: 'float64' for col in X_test.select_dtypes('int').columns})
    X_train_float = X_train.astype({col: 'float64' for col in X_train.select_dtypes('int').columns})
    
    # Create input example and signature
    input_example = X_test_float.iloc[[0]]
    signature = infer_signature(X_train_float, best_model.predict(X_train_float))
    
    # Log the model with signature and input example
    mlflow.sklearn.log_model(
        sk_model=best_model, 
        artifact_path="enhanced_multinomial_regression_model",
        input_example=input_example,
        signature=signature
    )
    
    print("Enhanced Multinomial Regression model logged to MLflow successfully!")
    print(f"Run ID: {mlflow.active_run().info.run_id}")
    print(f"Model artifact path: enhanced_multinomial_regression_model")
    
    # Register model to MLflow Model Registry
    model_name = "WineQuality_Multinomial_Regression"
    try:
        model_version = mlflow.register_model(
            model_uri=f"runs:/{mlflow.active_run().info.run_id}/enhanced_multinomial_regression_model",
            name=model_name
        )

        print(f"Model registered to MLflow Model Registry!")
        print(f"Model name: {model_name}")
        print(f"Model version: {model_version.version}")

        # Add model description and tags
        from mlflow.tracking import MlflowClient
        client = MlflowClient()

        # Update model version with description and tags
        client.update_model_version(
            name=model_name,
            version=model_version.version,
            description=(
                f"Enhanced Multinomial Logistic Regression for Wine Quality Classification. "
                f"Accuracy: {accuracy_best:.4f}, F1-weighted: {f1_best:.4f}. "
                f"Uses StandardScaler preprocessing and GridSearchCV hyperparameter tuning."
            )
        )

        # Set tags for the model version
        client.set_model_version_tag(
            name=model_name,
            version=model_version.version,
            key="model_type",
            value="multinomial_logistic_regression"
        )

        client.set_model_version_tag(
            name=model_name,
            version=model_version.version,
            key="preprocessing",
            value="StandardScaler"
        )

        client.set_model_version_tag(
            name=model_name,
            version=model_version.version,
            key="hyperparameter_tuning",
            value="GridSearchCV"
        )

        print("Model version updated with description and tags!")
    except Exception as e:
        # Catch registry/network/permission errors and continue — log for debugging
        print(f"Warning: could not register/update model in MLflow Model Registry: {e}")

# ============================================== COMPARISON WITH ORIGINAL MODEL ==============================================

print("\n" + "="*80)
print("COMPARISON WITH ORIGINAL MODEL")
print("="*80)

# Compare with the original model from Cell 16
print("=== MODEL COMPARISON ===")
print("Original Model (Cell 16):")
print(f"  Accuracy: 0.8389")
print(f"  No preprocessing")
print(f"  Basic hyperparameters")

print("\nEnhanced Model (Current):")
print(f"  Accuracy: {accuracy_best:.4f}")
print(f"  With StandardScaler preprocessing")
print(f"  Optimized hyperparameters via GridSearchCV")
print(f"  Cross-validation F1: {cv_scores.mean():.4f}")

improvement = ((accuracy_best - 0.8389) / 0.8389) * 100
print(f"\nImprovement: {improvement:.2f}%")


