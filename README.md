# 🛡️ Cybersecurity Threat Detection System

Machine Learning and Deep Learning implementation for cybersecurity threat detection based on the research paper:

**"Evaluating Predictive Models in Cybersecurity: A Comparative Analysis of Machine and Deep Learning Techniques for Threat Detection"**  
📄 Paper: https://arxiv.org/pdf/2407.06014

---

## 🎯 Project Overview

This project implements a comprehensive threat detection system comparing **Machine Learning** and **Deep Learning** approaches for cybersecurity:

### **Machine Learning Models**
- ✅ Random Forest
- ✅ Support Vector Machine (SVM)
- ✅ XGBoost
- ✅ Gradient Boosting

### **Deep Learning Models**
- ✅ CNN (Convolutional Neural Network)
- ✅ LSTM (Long Short-Term Memory)
- ✅ VGG (Visual Geometry Group)
- ✅ ResNet (Residual Network)

### **Key Enhancements Over Paper**
1. 🔄 **SMOTE & ADASYN** for class imbalance
2. 🎯 **Outlier detection** (Isolation Forest, Z-score)
3. 📊 **K-Fold Cross-Validation** (instead of single split)
4. 🧬 **Advanced feature engineering** (entropy, temporal features)
5. 🔍 **Model interpretability** (SHAP values)
6. 📈 **Hyperparameter optimization** (Optuna)
7. 🎨 **Interactive visualizations** (Plotly dashboards)

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/fadhly-git/cybersecurity-threat-detection.git
cd cybersecurity-threat-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements. txt
```

### Usage

#### 1️⃣ **Prepare Your Data**
```bash
# Place your datasets in data/raw/
data/raw/cybersecurity_attacks.csv
data/raw/wsn_dataset.csv
```

#### 2️⃣ **Train Models** ⭐ **NEW: Auto-Logging!**
```bash
# Train hybrid models (RECOMMENDED)
python scripts/train_all_models.py --models cnn_lstm_mlp --epochs 10

# Train all models
python scripts/train_all_models.py --models all --epochs 50

# Output saved to: logs/training/train_all_models_TIMESTAMP.log
```

#### 3️⃣ **Train Specific Model Types**
```bash
# Train ML models
python scripts/train_ml_models.py --models rf,xgb --cv 5

# Train DL models
python scripts/train_dl_models.py --models cnn,lstm --epochs 50
```

#### 4️⃣ **Evaluate Models**
```bash
python scripts/evaluate_models.py --models-dir results/models
```

📝 **All training output automatically logged to `logs/` directory!**  
See [LOGGING_QUICKSTART.md](LOGGING_QUICKSTART.md) for details.

---

## 📊 Expected Results

Based on paper benchmarks:

| Model | Dataset 1 Accuracy | Dataset 2 Accuracy |
|-------|-------------------|-------------------|
| Random Forest | 99.01% | 36.21% |
| SVM | 98.87% | 35.98% |
| XGBoost | **99.15%** | **37.45%** |
| CNN | 97.23% | 42.11% |
| LSTM | 96.78% | 45.32% |
| VGG | 98.12% | 48.67% |
| ResNet | 98.45% | **51.23%** |

*With enhancements, we expect to improve Dataset 2 performance significantly.*

---

## 📁 Project Structure

```
cybersecurity-threat-detection/
├── config/                 # Configuration files
├── data/                   # Data directory
├── src/                    # Source code
│   ├── data/              # Preprocessing & feature engineering
│   ├── models/            # ML & DL models
│   ├── evaluation/        # Metrics & visualization
│   └── utils/             # Helper functions
├── scripts/               # Execution scripts
├── notebooks/             # Jupyter notebooks
├── tests/                 # Unit tests
├── docs/                  # Documentation
└── results/               # Output results
```

---

## 📖 Documentation

- **[Data Preprocessing Guide](docs/data_preprocessing.md)** - Detailed preprocessing steps
- **[Model Architecture](docs/model_architecture.md)** - Model designs and hyperparameters
- **[Results Analysis](docs/results.md)** - Performance metrics and comparisons
- **[Usage Guide](docs/usage_guide.md)** - Advanced usage examples

---

## 🔬 Research Paper Citation

```bibtex
@article{cybersecurity2024,
  title={Evaluating Predictive Models in Cybersecurity: A Comparative Analysis of Machine and Deep Learning Techniques for Threat Detection},
  author={[Authors]},
  journal={arXiv preprint arXiv:2407.06014},
  year={2024}
}
```

---

## 🛠️ Technologies Used

- **Python 3.9+**
- **scikit-learn** - ML models
- **TensorFlow/Keras** - DL models
- **PyTorch** - Alternative DL framework
- **imbalanced-learn** - SMOTE/ADASYN
- **SHAP** - Model interpretability
- **Plotly** - Interactive visualizations
- **Optuna** - Hyperparameter optimization

---

## 📝 License

MIT License - see [LICENSE](LICENSE) file

---

## 👤 Author

**Fadhly**  
GitHub: [@fadhly-git](https://github.com/fadhly-git)

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! 

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 🙏 Acknowledgments

- Research paper authors for the methodology
- Open-source community for amazing tools
- Cybersecurity datasets providers

---

**⭐ If you find this project useful, please star the repository! **
