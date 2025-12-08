"""Helper utilities for the cybersecurity threat detection system.

This module provides various helper functions for data processing,
file operations, configuration management, and common tasks.
"""

import os
import pickle
import json
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime
import hashlib


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
    
    Returns:
        Configuration dictionary
    
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        output_path: Path to save YAML file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


def load_pickle(file_path: str) -> Any:
    """Load object from pickle file.
    
    Args:
        file_path: Path to pickle file
    
    Returns:
        Loaded object
    """
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def save_pickle(obj: Any, file_path: str) -> None:
    """Save object to pickle file.
    
    Args:
        obj: Object to save
        file_path: Path to save pickle file
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_json(file_path: str) -> Dict[str, Any]:
    """Load dictionary from JSON file.
    
    Args:
        file_path: Path to JSON file
    
    Returns:
        Dictionary loaded from JSON
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Dict[str, Any], file_path: str, indent: int = 2) -> None:
    """Save dictionary to JSON file.
    
    Args:
        data: Dictionary to save
        file_path: Path to save JSON file
        indent: JSON indentation level
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def ensure_dir(directory: Union[str, Path]) -> Path:
    """Create directory if it doesn't exist.
    
    Args:
        directory: Directory path
    
    Returns:
        Path object for the directory
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def get_timestamp() -> str:
    """Get current timestamp as formatted string.
    
    Returns:
        Timestamp string in format YYYYMMDD_HHMMSS
    """
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def get_file_hash(file_path: str, algorithm: str = 'md5') -> str:
    """Calculate hash of a file.
    
    Args:
        file_path: Path to file
        algorithm: Hash algorithm ('md5', 'sha256', etc.)
    
    Returns:
        Hexadecimal hash string
    """
    hash_func = hashlib.new(algorithm)
    
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hash_func.update(chunk)
    
    return hash_func.hexdigest()


def format_bytes(size: int) -> str:
    """Format byte size to human-readable string.
    
    Args:
        size: Size in bytes
    
    Returns:
        Formatted string (e.g., '1.5 GB')
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"


def format_time(seconds: float) -> str:
    """Format seconds to human-readable time string.
    
    Args:
        seconds: Time in seconds
    
    Returns:
        Formatted time string (e.g., '1h 23m 45s')
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}h {minutes}m {secs}s"


def merge_dicts(dict1: Dict, dict2: Dict, deep: bool = True) -> Dict:
    """Merge two dictionaries.
    
    Args:
        dict1: First dictionary
        dict2: Second dictionary (takes precedence)
        deep: Whether to perform deep merge
    
    Returns:
        Merged dictionary
    """
    if not deep:
        result = dict1.copy()
        result.update(dict2)
        return result
    
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value, deep=True)
        else:
            result[key] = value
    
    return result


def split_dataframe(
    df: pd.DataFrame,
    ratios: List[float],
    shuffle: bool = True,
    random_state: Optional[int] = None
) -> List[pd.DataFrame]:
    """Split dataframe into multiple parts.
    
    Args:
        df: DataFrame to split
        ratios: List of ratios (should sum to 1.0)
        shuffle: Whether to shuffle before splitting
        random_state: Random state for reproducibility
    
    Returns:
        List of DataFrames
    
    Example:
        >>> train, val, test = split_dataframe(df, [0.7, 0.15, 0.15])
    """
    assert abs(sum(ratios) - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    if shuffle:
        df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    splits = []
    start_idx = 0
    
    for i, ratio in enumerate(ratios[:-1]):
        split_size = int(len(df) * ratio)
        splits.append(df.iloc[start_idx:start_idx + split_size])
        start_idx += split_size
    
    # Last split gets remaining rows
    splits.append(df.iloc[start_idx:])
    
    return splits


def memory_usage(df: pd.DataFrame, detailed: bool = False) -> Union[str, pd.DataFrame]:
    """Get memory usage of DataFrame.
    
    Args:
        df: DataFrame to analyze
        detailed: Whether to return detailed per-column usage
    
    Returns:
        Memory usage string or detailed DataFrame
    """
    if detailed:
        mem_usage = df.memory_usage(deep=True)
        mem_df = pd.DataFrame({
            'Column': mem_usage.index,
            'Memory': mem_usage.values,
            'Memory (Human)': [format_bytes(x) for x in mem_usage.values]
        })
        return mem_df
    else:
        total_bytes = df.memory_usage(deep=True).sum()
        return format_bytes(total_bytes)


def downcast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns to reduce memory usage.
    
    Args:
        df: DataFrame to downcast
    
    Returns:
        DataFrame with downcasted numeric columns
    """
    df = df.copy()
    
    # Downcast integers
    int_cols = df.select_dtypes(include=['int']).columns
    for col in int_cols:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    
    # Downcast floats
    float_cols = df.select_dtypes(include=['float']).columns
    for col in float_cols:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    return df


def describe_data(df: pd.DataFrame) -> Dict[str, Any]:
    """Get comprehensive description of DataFrame.
    
    Args:
        df: DataFrame to describe
    
    Returns:
        Dictionary with various statistics
    """
    description = {
        'shape': df.shape,
        'n_rows': len(df),
        'n_columns': len(df.columns),
        'memory_usage': memory_usage(df),
        'dtypes': df.dtypes.value_counts().to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
        'duplicate_rows': df.duplicated().sum(),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist(),
    }
    
    return description


def print_dict(d: Dict, indent: int = 0) -> None:
    """Pretty print nested dictionary.
    
    Args:
        d: Dictionary to print
        indent: Indentation level
    """
    for key, value in d.items():
        print('  ' * indent + str(key) + ':', end=' ')
        if isinstance(value, dict):
            print()
            print_dict(value, indent + 1)
        else:
            print(value)


class Timer:
    """Context manager for timing code blocks.
    
    Example:
        >>> with Timer('Data loading'):
        >>>     df = pd.read_csv('data.csv')
        Data loading completed in 2.34s
    """
    
    def __init__(self, name: str = 'Operation', verbose: bool = True):
        """Initialize timer.
        
        Args:
            name: Name of the operation being timed
            verbose: Whether to print timing information
        """
        self.name = name
        self.verbose = verbose
        self.start_time = None
        self.elapsed = None
    
    def __enter__(self):
        """Start timing."""
        self.start_time = datetime.now()
        if self.verbose:
            print(f"{self.name} started...")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing and print result."""
        end_time = datetime.now()
        self.elapsed = (end_time - self.start_time).total_seconds()
        
        if self.verbose:
            if exc_type is None:
                print(f"{self.name} completed in {format_time(self.elapsed)}")
            else:
                print(f"{self.name} failed after {format_time(self.elapsed)}")
