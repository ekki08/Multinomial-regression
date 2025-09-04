"""
MLflow Integration untuk Multinomial Logistic Regression - Wine Quality Classification
Script ini fokus pada logging model multinomial regression ke MLflow dengan konfigurasi yang optimal.
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           classification_report, confusion_matrix)
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load dan persiapkan data wine quality"""
    print("Loading dan mempersiapkan data...")
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
    
    # Prepare features and target
    X = df.drop(columns=["quality", "qualityg"])
    y = df["qualityg"]
    
    print(f"Dataset shape: {df.shape}")
    print(f"Target distribution: {np.bincount(y)}")
    print(f"Features: {list(X.columns)}")
    
    return X, y

def create_and_tune_model(X, y):
    """Buat dan tune model multinomial regression"""
    print("Membuat dan melakukan hyperparameter tuning...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create pipeline (removed deprecated multi_class parameter)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('multinomial_lr', LogisticRegression(
            solver='lbfgs', 
            random_state=42
        ))
    ])
    
    # Parameter grid untuk tuning
    param_grid = {
        'multinomial_lr__C': [0.1, 1, 10, 100],
        'multinomial_lr__max_iter': [500, 1000, 2000],
        'multinomial_lr__solver': ['lbfgs', 'newton-cg']
    }
    
    # GridSearchCV
    grid_search = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=5, 
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    # Predictions
    y_pred = best_model.predict(X_test)
    
    # Cross-validation
    cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='f1_weighted')
    
    return {
        'model': best_model,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred': y_pred,
        'cv_scores': cv_scores,
        'grid_search': grid_search
    }

def calculate_metrics(y_test, y_pred):
    """Hitung semua metrics yang diperlukan"""
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision_weighted': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall_weighted': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1_weighted': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'precision_per_class': precision_score(y_test, y_pred, average=None, zero_division=0),
        'recall_per_class': recall_score(y_test, y_pred, average=None, zero_division=0),
        'f1_per_class': f1_score(y_test, y_pred, average=None, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    return metrics

def log_to_mlflow(model_data, X, metrics):
    """Log model dan metrics ke MLflow"""
    print("Logging model ke MLflow...")
    
    with mlflow.start_run(run_name="Multinomial_Regression_MLflow"):
        # Extract data
        model = model_data['model']
        X_train = model_data['X_train']
        X_test = model_data['X_test']
        y_train = model_data['y_train']
        y_test = model_data['y_test']
        cv_scores = model_data['cv_scores']
        grid_search = model_data['grid_search']
        
        # Log parameters
        lr_params = model.named_steps['multinomial_lr']
        mlflow.log_param("model_type", "multinomial_logistic_regression")
        mlflow.log_param("solver", lr_params.solver)
        mlflow.log_param("max_iter", lr_params.max_iter)
        mlflow.log_param("C", lr_params.C)
        mlflow.log_param("multi_class", "auto_multinomial")
        mlflow.log_param("preprocessing", "StandardScaler")
        mlflow.log_param("random_state", 42)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("tuning_method", "GridSearchCV")
        
        # Log best parameters from grid search
        for param, value in grid_search.best_params_.items():
            mlflow.log_param(f"best_{param}", value)
        
        # Log metrics
        mlflow.log_metric("accuracy", metrics['accuracy'])
        mlflow.log_metric("precision_weighted", metrics['precision_weighted'])
        mlflow.log_metric("recall_weighted", metrics['recall_weighted'])
        mlflow.log_metric("f1_weighted", metrics['f1_weighted'])
        mlflow.log_metric("cv_f1_mean", cv_scores.mean())
        mlflow.log_metric("cv_f1_std", cv_scores.std())
        mlflow.log_metric("best_cv_score", grid_search.best_score_)
        
        # Log class-wise metrics
        class_names = ['low', 'medium', 'high']
        for i, class_name in enumerate(class_names):
            mlflow.log_metric(f"precision_{class_name}", metrics['precision_per_class'][i])
            mlflow.log_metric(f"recall_{class_name}", metrics['recall_per_class'][i])
            mlflow.log_metric(f"f1_{class_name}", metrics['f1_per_class'][i])
        
        # Log feature importance
        feature_names = X.columns
        coefficients = lr_params.coef_
        overall_importance = np.abs(coefficients).mean(axis=0)
        feature_importance_dict = dict(zip(feature_names, overall_importance))
        mlflow.log_dict(feature_importance_dict, "feature_importance.json")
        
        # Log confusion matrix
        cm_df = pd.DataFrame(
            metrics['confusion_matrix'], 
            index=['Actual_Low', 'Actual_Medium', 'Actual_High'],
            columns=['Pred_Low', 'Pred_Medium', 'Pred_High']
        )
        mlflow.log_dict(cm_df.to_dict(), "confusion_matrix.json")
        
        # Prepare data untuk signature
        X_train_float = X_train.astype({col: 'float64' for col in X_train.select_dtypes('int').columns})
        X_test_float = X_test.astype({col: 'float64' for col in X_test.select_dtypes('int').columns})
        
        # Create input example dan signature
        input_example = X_test_float.iloc[[0]]
        signature = infer_signature(X_train_float, model.predict(X_train_float))
        
        # Log model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="multinomial_regression_model",
            input_example=input_example,
            signature=signature
        )
        
        # Model Registry
        model_name = "WineQuality_Multinomial_Regression"
        model_version = mlflow.register_model(
            model_uri=f"runs:/{mlflow.active_run().info.run_id}/multinomial_regression_model",
            name=model_name
        )
        
        # Update model dengan description dan tags (simplified to avoid YAML issues)
        client = MlflowClient()
        try:
            client.update_model_version(
                name=model_name,
                version=model_version.version,
                description=f"Multinomial Logistic Regression untuk Wine Quality Classification. "
                           f"Accuracy: {metrics['accuracy']:.4f}, F1-weighted: {metrics['f1_weighted']:.4f}. "
                           f"Menggunakan StandardScaler preprocessing dan GridSearchCV hyperparameter tuning."
            )
            
            # Set tags
            tags = {
                "model_type": "multinomial_logistic_regression",
                "preprocessing": "StandardScaler",
                "hyperparameter_tuning": "GridSearchCV",
                "dataset": "wine_quality_red",
                "target_classes": "3_classes_low_medium_high"
            }
            
            for key, value in tags.items():
                client.set_model_version_tag(
                    name=model_name,
                    version=model_version.version,
                    key=key,
                    value=value
                )
            print("✅ Model description dan tags berhasil diupdate!")
        except Exception as e:
            print(f"⚠️  Warning: Gagal mengupdate model description/tags: {e}")
            print("Model tetap berhasil diregistrasi!")
        
        print(f"✅ Model berhasil di-log ke MLflow!")
        print(f"📊 Run ID: {mlflow.active_run().info.run_id}")
        print(f"🏷️  Model Name: {model_name}")
        print(f"📋 Model Version: {model_version.version}")
        print(f"🎯 Accuracy: {metrics['accuracy']:.4f}")
        print(f"📈 F1-weighted: {metrics['f1_weighted']:.4f}")
        
        return mlflow.active_run().info.run_id, model_version

def main():
    """Main function untuk menjalankan seluruh pipeline"""
    print("="*80)
    print("MULTINOMIAL REGRESSION - MLFLOW INTEGRATION")
    print("="*80)
    
    # Load dan prepare data
    X, y = load_and_prepare_data()
    
    # Create dan tune model
    model_data = create_and_tune_model(X, y)
    
    # Calculate metrics
    metrics = calculate_metrics(model_data['y_test'], model_data['y_pred'])
    
    # Log ke MLflow
    run_id, model_version = log_to_mlflow(model_data, X, metrics)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"✅ Model training completed!")
    print(f"📊 Best parameters: {model_data['grid_search'].best_params_}")
    print(f"🎯 Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"📈 Test F1-weighted: {metrics['f1_weighted']:.4f}")
    print(f"📊 CV F1 mean: {model_data['cv_scores'].mean():.4f} (±{model_data['cv_scores'].std():.4f})")
    print(f"🏷️  MLflow Run ID: {run_id}")
    print(f"📋 Model Version: {model_version.version}")
    print("="*80)

if __name__ == "__main__":
    main()
