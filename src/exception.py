import sys
import logging
from src.logger import get_logger

logger = get_logger()

class CustomException(Exception):
    def __init__(self, message, errors=None):
        super().__init__(message)
        _, _, tb = sys.exc_info()
        if tb:
            self.filename = tb.tb_frame.f_code.co_filename
            self.lineno = tb.tb_lineno
        else:
            self.filename = "Unknown"
            self.lineno = -1
        self.errors = errors
        logger.error(f"Exception occurred in {self.filename} at line {self.lineno}: {message}")

    def __str__(self):
        return f"{self.__class__.__name__}: {self.args[0]} (File: {self.filename}, Line: {self.lineno})"
