"""Logging configuration for the cybersecurity threat detection system.

This module provides comprehensive logging utilities with file and console output,
custom formatting, and support for different log levels.
"""

import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    log_format: Optional[str] = None,
    console: bool = True,
) -> logging.Logger:
    """Configure and return a logger instance.
    
    Args:
        name: Name of the logger
        log_file: Path to log file. If None, uses default location
        level: Logging level (e.g., logging.INFO, logging.DEBUG)
        log_format: Custom format string. If None, uses default format
        console: Whether to also log to console
    
    Returns:
        Configured logger instance
    
    Example:
        >>> logger = setup_logger('preprocessing', level=logging.DEBUG)
        >>> logger.info('Starting data preprocessing')
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers = []
    
    # Default format
    if log_format is None:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    formatter = logging.Formatter(log_format, datefmt='%Y-%m-%d %H:%M:%S')
    
    # File handler
    if log_file is None:
        # Create logs directory if it doesn't exist
        log_dir = Path('logs')
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log file with timestamp
        timestamp = datetime.now().strftime('%Y%m%d')
        log_file = log_dir / f'{name}_{timestamp}.log'
    
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get an existing logger or create a new one with default settings.
    
    Args:
        name: Name of the logger
    
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    
    # If logger has no handlers, set it up with defaults
    if not logger.handlers:
        return setup_logger(name)
    
    return logger


class DualOutput:
    """Redirect stdout/stderr to both console and file.
    
    Usage:
        with DualOutput('logs/training.log'):
            print('This goes to both console and file')
    """
    def __init__(self, log_file: str, mode: str = 'a'):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.terminal = sys.stdout
        self.file = None
        self._progress_buffer = ''
        self._ansi_pattern = re.compile(r"\x1B[@-_][0-?]*[ -/]*[@-~]")
        
    def write(self, message):
        """Write to both terminal and file."""
        self.terminal.write(message)
        if self.file:
            # Handle carriage return progress updates (in-place updates)
            if '\r' in message:
                # Split by \r to get all the progress updates
                parts = message.split('\r')
                
                # Only keep the last part (most recent state)
                # If message ends with \n, that's the final state - write it
                if message.endswith('\n'):
                    # Last update before newline - write it
                    final_part = parts[-1] if parts[-1] else parts[-2] if len(parts) > 1 else ''
                    cleaned = self._ansi_pattern.sub('', final_part)
                    if cleaned.strip():
                        self.file.write(cleaned)
                    self._progress_buffer = ''
                else:
                    # Still updating - just buffer the latest state, don't write
                    self._progress_buffer = parts[-1]
                return
            
            # Regular message without \r - write it normally
            cleaned_message = self._ansi_pattern.sub('', message)
            if cleaned_message:
                self.file.write(cleaned_message)
            self.file.flush()
    
    def flush(self):
        """Flush both outputs."""
        self.terminal.flush()
        if self.file:
            if self._progress_buffer:
                cleaned_buffer = self._progress_buffer.replace("\r", "")
                if cleaned_buffer:
                    self.file.write(cleaned_buffer)
                self._progress_buffer = ''
            self.file.flush()
    
    def __enter__(self):
        """Start dual output."""
        self.file = open(self.log_file, self.mode, encoding='utf-8')
        sys.stdout = self
        sys.stderr = self
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore original output."""
        sys.stdout = self.terminal
        sys.stderr = self.terminal
        if self.file:
            self.file.close()


class LoggerMixin:
    """Mixin class to add logging capabilities to any class.
    
    Usage:
        class MyClass(LoggerMixin):
            def __init__(self):
                super().__init__()
                self.logger.info('MyClass initialized')
    """
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        name = f"{self.__class__.__module__}.{self.__class__.__name__}"
        return get_logger(name)


def log_function_call(func):
    """Decorator to log function calls with arguments and execution time.
    
    Args:
        func: Function to decorate
    
    Returns:
        Wrapped function
    
    Example:
        @log_function_call
        def process_data(data):
            # Function implementation
            pass
    """
    import functools
    import time
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        
        # Log function call
        func_name = func.__name__
        logger.debug(f"Calling {func_name} with args={args}, kwargs={kwargs}")
        
        # Execute function and measure time
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.debug(f"{func_name} completed in {elapsed:.2f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"{func_name} failed after {elapsed:.2f}s: {str(e)}")
            raise
    
    return wrapper


def configure_root_logger(
    level: int = logging.INFO,
    log_file: str = 'logs/application.log'
) -> None:
    """Configure the root logger for the entire application.
    
    Args:
        level: Logging level
        log_file: Path to main log file
    """
    # Create logs directory
    log_dir = Path(log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file, mode='a', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )


# Module-level logger
_module_logger = None


def get_module_logger() -> logging.Logger:
    """Get module-level logger."""
    global _module_logger
    if _module_logger is None:
        _module_logger = setup_logger('cybersecurity_detection')
    return _module_logger
