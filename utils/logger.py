import logging
import sys
from datetime import datetime
from pathlib import Path

class Logger:
    def __init__(self, name="ai_video_generator", level=logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        if not self.logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
            
            # File handler
            log_file = Path("logs") / f"video_generator_{datetime.now().strftime('%Y%m%d')}.log"
            log_file.parent.mkdir(exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(console_formatter)
            self.logger.addHandler(file_handler)
    
    def info(self, message):
        self.logger.info(message)
    
    def error(self, message):
        self.logger.error(message)
    
    def warning(self, message):
        self.logger.warning(message)
    
    def debug(self, message):
        self.logger.debug(message)
