# MLflow Integration untuk Multinomial Regression

## Overview
Dokumen ini menjelaskan cara mengintegrasikan model multinomial logistic regression dengan MLflow untuk tracking dan model registry.

## File yang Tersedia

### 1. `multinomial_regression_enhanced.py`
File asli yang sudah diupgrade dengan integrasi MLflow lengkap:
- ✅ Comprehensive logging (parameters, metrics, artifacts)
- ✅ Model registry integration
- ✅ Feature importance logging
- ✅ Confusion matrix logging
- ✅ Cross-validation metrics
- ✅ Class-wise performance metrics

### 2. `multinomial_mlflow.py`
Script terpisah yang fokus pada integrasi MLflow:
- ✅ Clean, modular code structure
- ✅ Error handling untuk model registry
- ✅ Comprehensive logging
- ✅ User-friendly output dengan emoji
- ✅ Function-based approach

## Cara Menjalankan

### Opsi 1: Jalankan Enhanced Script (Lengkap)
```bash
python multinomial_regression_enhanced.py
```

### Opsi 2: Jalankan MLflow-focused Script (Bersih)
```bash
python multinomial_mlflow.py
```

### Lihat MLflow UI
```bash
mlflow ui
```
Kemudian buka browser ke `http://localhost:5000`

## Fitur-Fitur MLflow yang Diimplementasikan

### 1. **Parameter Logging**
- Model type: multinomial_logistic_regression
- Solver: lbfgs/newton-cg
- C (regularization): 0.1, 1, 10, 100
- Max iterations: 500, 1000, 2000
- Preprocessing: StandardScaler
- Hyperparameter tuning method: GridSearchCV

### 2. **Metrics Logging**
- Accuracy
- Precision (weighted & per-class)
- Recall (weighted & per-class) 
- F1-score (weighted & per-class)
- Cross-validation metrics (mean & std)
- Best CV score dari GridSearch

### 3. **Artifacts Logging**
- Feature importance (JSON)
- Confusion matrix (JSON)
- Model dengan signature dan input example

### 4. **Model Registry**
- Model name: `WineQuality_Multinomial_Regression`
- Automatic versioning
- Model description dengan performance metrics
- Tags untuk metadata

## Hasil Performance

### Model Metrics (Terbaru)
- **Accuracy**: 83.75%
- **F1-weighted**: 80.65%
- **CV F1 mean**: 79.79% (±1.65%)

### Best Parameters
- **C**: 0.1
- **max_iter**: 500
- **solver**: lbfgs

### Class-wise Performance
- **Low Quality** (0): Precision, Recall, F1-score
- **Medium Quality** (1): Precision, Recall, F1-score  
- **High Quality** (2): Precision, Recall, F1-score

## Perbaikan yang Dilakukan

### 1. **Deprecated Warning Fix**
- Menghilangkan parameter `multi_class='multinomial'` yang deprecated
- Sklearn versi terbaru otomatis menggunakan multinomial untuk multi-class

### 2. **Error Handling**
- Try-catch untuk model registry operations
- Graceful handling untuk YAML serialization issues
- Warning messages yang informatif

### 3. **Enhanced Logging**
- Signature inference untuk model deployment
- Input example untuk testing
- Comprehensive parameter dan metrics logging

### 4. **User Experience**
- Progress indicators dengan emoji
- Clear output messages
- Summary section dengan key metrics

## Struktur MLflow Runs

```
mlruns/
├── 0/                          # Default experiment
│   ├── <run_id>/              # Individual run
│   │   ├── artifacts/         # Model artifacts
│   │   ├── metrics/           # Performance metrics
│   │   ├── params/            # Model parameters
│   │   └── tags/              # Run metadata
│   └── models/                # Registered models
│       └── WineQuality_Multinomial_Regression/
```

## Troubleshooting

### Common Issues

1. **YAML Serialization Error**
   - Fixed dengan error handling di model registry
   - Model tetap tersimpan meski description gagal

2. **Deprecated Warning**
   - Fixed dengan menghilangkan parameter `multi_class`
   - Sklearn otomatis detect multinomial untuk multi-class

3. **Model Registry Access**
   - Pastikan MLflow server berjalan
   - Check permissions untuk directory mlruns

## Next Steps

1. **Model Deployment**
   - Gunakan mlflow models serve
   - Deploy ke production environment

2. **A/B Testing**
   - Compare dengan model lain
   - Track performance over time

3. **Model Monitoring**
   - Set up alerts untuk performance degradation
   - Monitor data drift

## Kesimpulan

Integrasi MLflow berhasil diimplementasikan dengan fitur-fitur:
- ✅ Comprehensive experiment tracking
- ✅ Model registry dengan versioning
- ✅ Artifact storage untuk reproducibility
- ✅ Clean, maintainable code structure
- ✅ Error handling dan user-friendly output

Model multinomial regression sekarang sudah fully integrated dengan MLflow ecosystem!
