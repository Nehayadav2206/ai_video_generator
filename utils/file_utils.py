import os
import shutil
from pathlib import Path
from typing import List
import hashlib

class FileUtils:
    @staticmethod
    def ensure_directory(path: str) -> Path:
        """Ensure directory exists, create if not"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @staticmethod
    def clean_filename(filename: str) -> str:
        """Clean filename for safe saving"""
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        return filename[:100]  # Limit length
    
    @staticmethod
    def get_file_hash(filepath: str) -> str:
        """Get MD5 hash of file"""
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    @staticmethod
    def cleanup_temp_files(temp_dir: str):
        """Clean up temporary files"""
        temp_path = Path(temp_dir)
        if temp_path.exists():
            shutil.rmtree(temp_path)
            temp_path.mkdir()