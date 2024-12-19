# src/llm/format_validator.py

import re
from typing import Optional

class ResponseValidator:
    """Validates and cleans LLM responses"""
    
    @staticmethod
    def validate_format(response: str) -> bool:
        """Check if response follows required format"""
        # Remove empty lines and any lines starting with #
        lines = [l.strip() for l in response.split('\n') if l.strip() and not l.startswith('#')]
        
        # Check each line matches our format
        pattern = r'^[^()[\]]+\([^()]+\)\s*\[[^\[\]]+\]$'
        return all(re.match(pattern, line) for line in lines)
    
    @staticmethod
    def clean_response(response: str) -> str:
        """Clean response to match required format"""
        # Remove any lines that don't match our format
        pattern = r'^[^()[\]]+\([^()]+\)\s*\[[^\[\]]+\]$'
        lines = [l.strip() for l in response.split('\n')]
        valid_lines = [l for l in lines if l and re.match(pattern, l)]
        
        return '\n'.join(valid_lines)
    
    @staticmethod
    def validate_and_clean(response: str) -> Optional[str]:
        """Validate response and clean if necessary"""
        if not response:
            return None
            
        # First try to clean the response
        cleaned = ResponseValidator.clean_response(response)
        
        # Validate the cleaned response
        if ResponseValidator.validate_format(cleaned):
            return cleaned
            
        return None