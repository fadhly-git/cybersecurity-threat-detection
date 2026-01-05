"""
helpers.py - Helper utilities
"""

import time
import logging
import os
import random
import numpy as np
import platform
from datetime import datetime
from pathlib import Path
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import Config


class Timer:
    """Simple timer for measuring execution time"""
    
    def __init__(self, name: str = "Timer"):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.elapsed = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time
        print(f"⏱️  {self.name}: {self.elapsed:.2f} seconds")
        
    def start(self):
        self.start_time = time.time()
        
    def stop(self):
        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time
        return self.elapsed
    
    def get_elapsed(self) -> float:
        if self.start_time is None:
            return 0
        return time.time() - self.start_time


class Logger:
    """Simple logger"""
    
    def __init__(self, 
                 name: str = "CyberSecurityML",
                 log_dir: Optional[str] = None,
                 level: str = "INFO"):
        
        self.log_dir = Path(log_dir) if log_dir else Config.LOG_DIR
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Clear existing handlers
        self.logger.handlers = []
        
        # Create formatters
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        log_file = self.log_dir / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
    def info(self, message: str):
        self.logger.info(message)
        
    def debug(self, message: str):
        self.logger.debug(message)
        
    def warning(self, message: str):
        self.logger.warning(message)
        
    def error(self, message: str):
        self.logger.error(message)
        
    def critical(self, message: str):
        self.logger.critical(message)


def set_seeds(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
    
    print(f"🌱 Random seed set to: {seed}")


def print_system_info():
    """Print system information"""
    print("="*60)
    print("SYSTEM INFORMATION")
    print("="*60)
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Python: {platform.python_version()}")
    print(f"Processor: {platform.processor()}")
    print(f"Machine: {platform.machine()}")
    
    # Check available libraries
    print("\n📦 Libraries:")
    
    try:
        import numpy
        print(f"  NumPy: {numpy.__version__}")
    except ImportError:
        print("  NumPy: Not installed")
    
    try:
        import pandas
        print(f"  Pandas: {pandas.__version__}")
    except ImportError:
        print("  Pandas: Not installed")
    
    try:
        import sklearn
        print(f"  Scikit-learn: {sklearn.__version__}")
    except ImportError:
        print("  Scikit-learn: Not installed")
    
    try:
        import tensorflow
        print(f"  TensorFlow: {tensorflow.__version__}")
    except ImportError:
        print("  TensorFlow: Not installed")
    
    try:
        import xgboost
        print(f"  XGBoost: {xgboost.__version__}")
    except ImportError:
        print("  XGBoost: Not installed")
    
    try:
        import lightgbm
        print(f"  LightGBM: {lightgbm.__version__}")
    except ImportError:
        print("  LightGBM: Not installed")
    
    try:
        import catboost
        print(f"  CatBoost: {catboost.__version__}")
    except ImportError:
        print("  CatBoost: Not installed")
    
    print("="*60)


def format_time(seconds: float) -> str:
    """Format seconds to human readable string"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{int(minutes)}m {int(secs)}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{int(hours)}h {int(minutes)}m"


def get_memory_usage() -> str:
    """Get current memory usage"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem = process.memory_info().rss / 1024 / 1024  # MB
        return f"{mem:.1f} MB"
    except ImportError:
        return "N/A (psutil not installed)"