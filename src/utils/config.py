# src/utils/config.py

import yaml
import os
from pathlib import Path
import logging
from typing import Dict, Any

class ConfigLoader:
    """Configuration loader for the project"""
    
    def __init__(self, config_path: str = None):
        """
        Initialize config loader
        
        Args:
            config_path: Path to config.yaml. If None, uses default project location
        """
        if config_path is None:
            # Get project root directory (2 levels up from this file)
            project_root = Path(__file__).parent.parent.parent
            config_path = os.path.join(project_root, 'config', 'config.yaml')
            
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from yaml file"""
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            return self.config
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found at {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing config file: {str(e)}")

    def get_config(self) -> Dict[str, Any]:
        """Get full configuration"""
        return self.config        
        
    def get_gcp_config(self) -> Dict[str, str]:
        """Get Google Cloud Platform configuration"""
        return self.config['gcp']
    
    def get_data_generation_config(self) -> Dict[str, Any]:
        """Get data generation configuration"""
        return self.config['data_generation']
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        return self.config['model']
    
    def setup_logging(self):
        """Setup logging based on configuration"""
        logging_config = self.config['logging']
        
        # Create logs directory if it doesn't exist
        log_path = logging_config['file']
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging_config['level'],
            format=logging_config['format'],
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler()  # Also log to console
            ]
        )