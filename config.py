"""
config.py - Configuration for Cybersecurity Threat Detection
Optimized for macOS CPU
"""

import os
from pathlib import Path
from datetime import datetime


class Config:
    """Main configuration class"""
    
    # =========================================================================
    # PATHS
    # =========================================================================
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    MODEL_DIR = BASE_DIR / "saved_models"
    LOG_DIR = BASE_DIR / "logs"
    RESULTS_DIR = BASE_DIR / "results"
    
    # Create directories
    for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, 
                     MODEL_DIR, LOG_DIR, RESULTS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # DATASET CONFIGURATION
    # =========================================================================
    DATASET_NAME = "NSL-KDD"
    
    # NSL-KDD URLs
    NSL_KDD_TRAIN_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt"
    NSL_KDD_TEST_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest+.txt"
    
    # NSL-KDD Column names
    NSL_KDD_COLUMNS = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
        'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
        'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
        'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
        'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
        'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
        'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
        'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
        'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty'
    ]
    
    # Attack categories mapping
    ATTACK_CATEGORIES = {
        'normal': 'Normal',
        # DoS attacks
        'back': 'DoS', 'land': 'DoS', 'neptune': 'DoS', 'pod': 'DoS',
        'smurf': 'DoS', 'teardrop': 'DoS', 'apache2': 'DoS', 'udpstorm': 'DoS',
        'processtable': 'DoS', 'mailbomb': 'DoS',
        # Probe attacks
        'satan': 'Probe', 'ipsweep': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe',
        'mscan': 'Probe', 'saint': 'Probe',
        # R2L attacks
        'guess_passwd': 'R2L', 'ftp_write': 'R2L', 'imap': 'R2L', 'phf': 'R2L',
        'multihop': 'R2L', 'warezmaster': 'R2L', 'warezclient': 'R2L', 'spy': 'R2L',
        'xlock': 'R2L', 'xsnoop': 'R2L', 'snmpguess': 'R2L', 'snmpgetattack': 'R2L',
        'httptunnel': 'R2L', 'sendmail': 'R2L', 'named': 'R2L', 'worm': 'R2L',
        # U2R attacks
        'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'rootkit': 'U2R', 'perl': 'U2R',
        'sqlattack': 'U2R', 'xterm': 'U2R', 'ps': 'U2R', 'httptunnel': 'U2R'
    }
    
    # =========================================================================
    # DATA SPLIT CONFIGURATION
    # =========================================================================
    TEST_SIZE = 0.2
    VAL_SIZE = 0.1
    RANDOM_STATE = 42
    STRATIFIED = True
    
    # =========================================================================
    # TRAINING CONFIGURATION - CPU OPTIMIZED
    # =========================================================================
    BATCH_SIZE = 128          # Smaller batch for CPU
    EPOCHS = 50               # Reasonable for CPU
    EARLY_STOPPING_PATIENCE = 10
    LEARNING_RATE = 0.001
    
    # Cross Validation
    N_FOLDS = 5
    
    # =========================================================================
    # HYPERPARAMETER TUNING - CPU OPTIMIZED
    # =========================================================================
    N_TRIALS = 30             # Reduced for faster tuning
    TIMEOUT = 1800            # 30 minutes max
    
    # =========================================================================
    # MODEL SELECTION
    # =========================================================================
    USE_ML_MODELS = True      # ML models - fast on CPU
    USE_DL_MODELS = True      # DL models - slower but works
    USE_ENSEMBLE = True       # Ensemble models
    USE_ATTENTION = False     # Disable attention - too slow on CPU
    
    # =========================================================================
    # FEATURE ENGINEERING
    # =========================================================================
    USE_FEATURE_SELECTION = True
    FEATURE_SELECTION_METHOD = 'mutual_info'  # 'mutual_info', 'chi2', 'f_classif'
    MAX_FEATURES = 40
    
    # =========================================================================
    # IMBALANCED DATA HANDLING
    # =========================================================================
    USE_SMOTE = True
    SMOTE_SAMPLING_STRATEGY = 'auto'
    SMOTE_K_NEIGHBORS = 5
    
    # =========================================================================
    # DEVICE CONFIGURATION - CPU ONLY
    # =========================================================================
    USE_GPU = False           # Force CPU
    GPU_ID = -1               # No GPU
    
    # Parallel Processing
    N_JOBS = -1               # Use all CPU cores
    
    # =========================================================================
    # LOGGING
    # =========================================================================
    VERBOSE = 1
    LOG_LEVEL = "INFO"
    SAVE_MODELS = True
    SAVE_RESULTS = True


class ModelConfig:
    """Model-specific configurations"""
    
    # =========================================================================
    # RANDOM FOREST
    # =========================================================================
    RF_PARAMS = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'],
        'class_weight': ['balanced', None]
    }
    
    RF_DEFAULT = {
        'n_estimators': 200,
        'max_depth': 20,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'max_features': 'sqrt',
        'class_weight': 'balanced',
        'n_jobs': -1,
        'random_state': 42
    }
    
    # =========================================================================
    # XGBOOST
    # =========================================================================
    XGB_PARAMS = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.1, 1],
        'reg_lambda': [0, 0.1, 1]
    }
    
    XGB_DEFAULT = {
        'n_estimators': 200,
        'max_depth': 7,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'n_jobs': -1,
        'random_state': 42,
        'verbosity': 0
    }
    
    # =========================================================================
    # LIGHTGBM
    # =========================================================================
    LGBM_PARAMS = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 7, 10, -1],
        'learning_rate': [0.01, 0.05, 0.1],
        'num_leaves': [31, 50, 100],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
    }
    
    LGBM_DEFAULT = {
        'n_estimators': 200,
        'max_depth': 7,
        'learning_rate': 0.1,
        'num_leaves': 50,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'n_jobs': -1,
        'random_state': 42,
        'verbose': -1
    }
    
    # =========================================================================
    # CATBOOST
    # =========================================================================
    CATBOOST_PARAMS = {
        'iterations': [100, 200, 300],
        'depth': [4, 6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        'l2_leaf_reg': [1, 3, 5, 7]
    }
    
    CATBOOST_DEFAULT = {
        'iterations': 200,
        'depth': 6,
        'learning_rate': 0.1,
        'l2_leaf_reg': 3,
        'random_state': 42,
        'verbose': False
    }
    
    # =========================================================================
    # DEEP NEURAL NETWORK - CPU OPTIMIZED
    # =========================================================================
    DNN_PARAMS = {
        'hidden_layers': [(256, 128, 64), (128, 64, 32), (512, 256, 128)],
        'dropout_rate': [0.3, 0.4, 0.5],
        'activation': ['relu', 'elu'],
        'batch_norm': [True, False]
    }
    
    DNN_DEFAULT = {
        'hidden_layers': (256, 128, 64),
        'dropout_rate': 0.4,
        'activation': 'relu',
        'batch_norm': True
    }
    
    # =========================================================================
    # CNN - CPU OPTIMIZED (Lighter)
    # =========================================================================
    CNN_PARAMS = {
        'filters': [(32, 64), (64, 128), (32, 64, 128)],
        'kernel_size': [3, 5],
        'dropout_rate': [0.3, 0.4, 0.5]
    }
    
    CNN_DEFAULT = {
        'filters': (32, 64),
        'kernel_size': 3,
        'dropout_rate': 0.4
    }
    
    # =========================================================================
    # LSTM - CPU OPTIMIZED (Lighter)
    # =========================================================================
    LSTM_PARAMS = {
        'lstm_units': [(64, 32), (128, 64), (64,)],
        'dropout_rate': [0.3, 0.4, 0.5],
        'recurrent_dropout': [0.1, 0.2]
    }
    
    LSTM_DEFAULT = {
        'lstm_units': (64, 32),
        'dropout_rate': 0.4,
        'recurrent_dropout': 0.1
    }


class MetricsConfig:
    """Metrics configuration"""
    
    PRIMARY_METRIC = 'f1_weighted'
    
    METRICS_LIST = [
        'accuracy',
        'precision_weighted',
        'recall_weighted',
        'f1_weighted',
        'roc_auc_ovr',
        'cohen_kappa',
        'mcc'
    ]
    
    CLASSIFICATION_REPORT = True
    CONFUSION_MATRIX = True
    ROC_CURVE = True
    PR_CURVE = True