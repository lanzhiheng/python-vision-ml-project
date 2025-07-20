#!/usr/bin/env python3
"""
Configuration Manager Module

This module handles configuration loading and management for the Image Vector SDK,
including environment variables and YAML configuration files.
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path
import re

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Configuration manager that handles environment variables and YAML config files.
    """
    
    def __init__(self, 
                 config_file: Optional[Union[str, Path]] = None,
                 env_file: Optional[Union[str, Path]] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_file: Path to YAML configuration file
            env_file: Path to .env file
        """
        self.config_file = Path(config_file) if config_file else None
        self.env_file = Path(env_file) if env_file else None
        
        self.config = {}
        self.env_vars = {}
        
        # Load configurations
        self._load_env_file()
        self._load_config_file()
        
        logger.info("ConfigManager initialized")
    
    def _load_env_file(self):
        """Load environment variables from .env file."""
        if self.env_file and self.env_file.exists():
            try:
                with open(self.env_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            key = key.strip()
                            value = value.strip().strip('"\'')
                            self.env_vars[key] = value
                            os.environ.setdefault(key, value)
                
                logger.info(f"Loaded {len(self.env_vars)} environment variables from {self.env_file}")
            except Exception as e:
                logger.error(f"Failed to load env file {self.env_file}: {e}")
    
    def _load_config_file(self):
        """Load configuration from YAML file."""
        if self.config_file and self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    raw_config = yaml.safe_load(f)
                
                # Process environment variable substitutions
                self.config = self._process_env_substitutions(raw_config)
                
                logger.info(f"Loaded configuration from {self.config_file}")
            except Exception as e:
                logger.error(f"Failed to load config file {self.config_file}: {e}")
    
    def _process_env_substitutions(self, obj: Any) -> Any:
        """
        Process environment variable substitutions in configuration.
        
        Supports syntax: ${VAR_NAME:default_value}
        """
        if isinstance(obj, dict):
            return {key: self._process_env_substitutions(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._process_env_substitutions(item) for item in obj]
        elif isinstance(obj, str):
            return self._substitute_env_vars(obj)
        else:
            return obj
    
    def _substitute_env_vars(self, value: str) -> Any:
        """Substitute environment variables in a string value."""
        # Pattern: ${VAR_NAME:default_value} or ${VAR_NAME}
        pattern = r'\$\{([^}]+)\}'
        
        def replace_var(match):
            var_expr = match.group(1)
            
            if ':' in var_expr:
                var_name, default_value = var_expr.split(':', 1)
            else:
                var_name, default_value = var_expr, ''
            
            # Get value from environment or use default
            env_value = os.environ.get(var_name, default_value)
            
            # Try to convert to appropriate type
            return self._convert_type(env_value)
        
        # If the entire string is a single substitution, return the converted value
        if re.fullmatch(pattern, value):
            return replace_var(re.match(pattern, value))
        
        # Otherwise, substitute within the string
        return re.sub(pattern, lambda m: str(replace_var(m)), value)
    
    def _convert_type(self, value: str) -> Any:
        """Convert string value to appropriate type."""
        if not isinstance(value, str):
            return value
        
        # Boolean conversion
        if value.lower() in ('true', 'yes', '1', 'on'):
            return True
        elif value.lower() in ('false', 'no', '0', 'off'):
            return False
        
        # Integer conversion
        try:
            if '.' not in value:
                return int(value)
        except ValueError:
            pass
        
        # Float conversion
        try:
            return float(value)
        except ValueError:
            pass
        
        # Return as string
        return value
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'milvus.host')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """
        Set configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'milvus.host')
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        # Navigate to the parent dictionary
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
    
    def get_milvus_config(self) -> Dict[str, Any]:
        """Get Milvus configuration."""
        return {
            'host': self.get('milvus.host', 'localhost'),
            'port': self.get('milvus.port', 19530),
            'user': self.get('milvus.user', ''),
            'password': self.get('milvus.password', ''),
            'database': self.get('milvus.database', 'default'),
            'timeout': self.get('milvus.timeout', 30),
            'alias': 'default'
        }
    
    def get_clip_config(self) -> Dict[str, Any]:
        """Get CLIP model configuration."""
        return {
            'model_name': self.get('clip.model_name', 'ViT-B/32'),
            'pretrained': self.get('clip.pretrained', 'openai'),
            'device': self.get('clip.device', 'auto'),
            'image_size': self.get('clip.image_size', 224)
        }
    
    def get_vector_config(self) -> Dict[str, Any]:
        """Get vector configuration."""
        return {
            'dimension': self.get('vector.dimension', 512),
            'normalize': self.get('vector.normalize', True),
            'metric_type': self.get('milvus.metric_type', 'L2'),
            'index_type': self.get('milvus.index_type', 'IVF_FLAT'),
            'nlist': self.get('milvus.nlist', 1024)
        }
    
    def get_image_processing_config(self) -> Dict[str, Any]:
        """Get image processing configuration."""
        return {
            'supported_formats': self.get('image_processing.supported_formats', 
                                        ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']),
            'max_size_mb': self.get('image_processing.max_size_mb', 10),
            'batch_size': self.get('image_processing.batch_size', 32),
            'auto_resize': self.get('image_processing.auto_resize', True),
            'maintain_aspect_ratio': self.get('image_processing.maintain_aspect_ratio', True)
        }
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance configuration."""
        return {
            'enable_cache': self.get('performance.enable_cache', True),
            'cache_size': self.get('performance.cache_size', 1000),
            'cache_ttl': self.get('performance.cache_ttl', 3600),
            'parallel_processing': self.get('performance.parallel_processing', True),
            'max_workers': self.get('performance.max_workers', 4)
        }
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return {
            'level': self.get('logging.level', 'INFO'),
            'format': self.get('logging.format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
            'file': self.get('logging.file', 'logs/image_vectorization.log'),
            'max_file_size': self.get('logging.max_file_size', '10MB'),
            'backup_count': self.get('logging.backup_count', 5)
        }
    
    def validate_config(self) -> Dict[str, list]:
        """
        Validate the configuration and return any errors.
        
        Returns:
            Dictionary with validation errors by category
        """
        errors = {
            'milvus': [],
            'clip': [],
            'vector': [],
            'image_processing': [],
            'general': []
        }
        
        # Validate Milvus config
        milvus_config = self.get_milvus_config()
        if not milvus_config['host']:
            errors['milvus'].append("Milvus host is required")
        if not isinstance(milvus_config['port'], int) or milvus_config['port'] <= 0:
            errors['milvus'].append("Milvus port must be a positive integer")
        
        # Validate CLIP config
        clip_config = self.get_clip_config()
        if not clip_config['model_name']:
            errors['clip'].append("CLIP model name is required")
        
        # Validate vector config
        vector_config = self.get_vector_config()
        if not isinstance(vector_config['dimension'], int) or vector_config['dimension'] <= 0:
            errors['vector'].append("Vector dimension must be a positive integer")
        
        # Validate image processing config
        img_config = self.get_image_processing_config()
        if not isinstance(img_config['max_size_mb'], (int, float)) or img_config['max_size_mb'] <= 0:
            errors['image_processing'].append("Max image size must be a positive number")
        
        # Remove empty error lists
        return {k: v for k, v in errors.items() if v}
    
    def to_dict(self) -> Dict[str, Any]:
        """Return the full configuration as a dictionary."""
        return self.config.copy()
    
    def save_config(self, output_file: Union[str, Path]):
        """Save current configuration to a YAML file."""
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Configuration saved to {output_file}")
    
    def reload(self):
        """Reload configuration from files."""
        self.config.clear()
        self.env_vars.clear()
        
        self._load_env_file()
        self._load_config_file()
        
        logger.info("Configuration reloaded")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the current configuration."""
        return {
            'config_file': str(self.config_file) if self.config_file else None,
            'env_file': str(self.env_file) if self.env_file else None,
            'config_keys': list(self.config.keys()),
            'env_vars_count': len(self.env_vars),
            'milvus_host': self.get('milvus.host'),
            'clip_model': self.get('clip.model_name'),
            'vector_dimension': self.get('vector.dimension'),
            'validation_errors': len(self.validate_config())
        }