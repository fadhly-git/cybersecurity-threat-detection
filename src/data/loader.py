"""Data loading utilities for cybersecurity datasets.

This module provides functions and classes for loading various cybersecurity
datasets in different formats.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from src.utils.logger import LoggerMixin
from src.utils.helpers import Timer, describe_data


class DataLoader(LoggerMixin):
    """Data loader for cybersecurity datasets."""
    
    def __init__(self):
        """Initialize DataLoader."""
        super().__init__()
    
    def load_dataset(
        self,
        file_path: str,
        file_format: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """Load dataset from file.
        
        Args:
            file_path: Path to dataset file
            file_format: File format ('csv', 'json', 'parquet'). Auto-detected if None
            **kwargs: Additional arguments to pass to pandas read function
        
        Returns:
            Loaded DataFrame
        
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is not supported
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        # Auto-detect format
        if file_format is None:
            file_format = file_path.suffix.lower().lstrip('.')
        
        self.logger.info(f"Loading dataset from {file_path}")
        
        with Timer(f"Loading {file_format.upper()} file", verbose=False) as timer:
            if file_format == 'csv':
                df = pd.read_csv(file_path, **kwargs)
            elif file_format == 'json':
                df = pd.read_json(file_path, **kwargs)
            elif file_format == 'parquet':
                df = pd.read_parquet(file_path, **kwargs)
            elif file_format == 'excel' or file_format in ['xlsx', 'xls']:
                df = pd.read_excel(file_path, **kwargs)
            else:
                raise ValueError(f"Unsupported file format: {file_format}")
        
        self.logger.info(f"Loaded dataset with shape {df.shape} in {timer.elapsed:.2f}s")
        
        # Log dataset description
        desc = describe_data(df)
        self.logger.info(f"Dataset info: {desc['n_rows']} rows, {desc['n_columns']} columns")
        self.logger.info(f"Memory usage: {desc['memory_usage']}")
        
        return df
    
    def load_from_config(
        self,
        config: Dict[str, Any],
        dataset_name: str
    ) -> Tuple[pd.DataFrame, str]:
        """Load dataset based on configuration.
        
        Args:
            config: Configuration dictionary
            dataset_name: Name of dataset in config
        
        Returns:
            Tuple of (DataFrame, target_column_name)
        
        Raises:
            ValueError: If dataset not found in config
        """
        datasets = config.get('data', {}).get('datasets', [])
        
        # Find dataset in config
        dataset_config = None
        for ds in datasets:
            if ds['name'] == dataset_name:
                dataset_config = ds
                break
        
        if dataset_config is None:
            raise ValueError(f"Dataset '{dataset_name}' not found in configuration")
        
        self.logger.info(f"Loading dataset: {dataset_config['description']}")
        
        df = self.load_dataset(dataset_config['path'])
        target_column = dataset_config['target_column']
        
        # Verify target column exists
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
        
        return df, target_column
    
    def save_dataset(
        self,
        df: pd.DataFrame,
        file_path: str,
        file_format: Optional[str] = None,
        **kwargs
    ) -> None:
        """Save dataset to file.
        
        Args:
            df: DataFrame to save
            file_path: Output file path
            file_format: File format ('csv', 'json', 'parquet'). Auto-detected if None
            **kwargs: Additional arguments to pass to pandas write function
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Auto-detect format
        if file_format is None:
            file_format = file_path.suffix.lower().lstrip('.')
        
        self.logger.info(f"Saving dataset to {file_path}")
        
        with Timer(f"Saving {file_format.upper()} file", verbose=False) as timer:
            if file_format == 'csv':
                df.to_csv(file_path, index=False, **kwargs)
            elif file_format == 'json':
                df.to_json(file_path, **kwargs)
            elif file_format == 'parquet':
                df.to_parquet(file_path, index=False, **kwargs)
            else:
                raise ValueError(f"Unsupported file format: {file_format}")
        
        self.logger.info(f"Saved dataset in {timer.elapsed:.2f}s")
    
    def load_arrays(
        self,
        directory: str,
        prefix: str = ''
    ) -> Dict[str, np.ndarray]:
        """Load preprocessed arrays from directory.
        
        Args:
            directory: Directory containing .npy files
            prefix: Optional prefix for array files
        
        Returns:
            Dictionary mapping array names to numpy arrays
        """
        directory = Path(directory)
        arrays = {}
        
        for file_path in directory.glob(f"{prefix}*.npy"):
            array_name = file_path.stem.replace(prefix, '')
            self.logger.info(f"Loading array: {array_name}")
            arrays[array_name] = np.load(file_path)
        
        self.logger.info(f"Loaded {len(arrays)} arrays from {directory}")
        return arrays
    
    def save_arrays(
        self,
        arrays: Dict[str, np.ndarray],
        directory: str,
        prefix: str = ''
    ) -> None:
        """Save arrays to directory.
        
        Args:
            arrays: Dictionary mapping array names to numpy arrays
            directory: Output directory
            prefix: Optional prefix for array files
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        for name, array in arrays.items():
            file_path = directory / f"{prefix}{name}.npy"
            self.logger.info(f"Saving array: {name} with shape {array.shape}")
            np.save(file_path, array)
        
        self.logger.info(f"Saved {len(arrays)} arrays to {directory}")
