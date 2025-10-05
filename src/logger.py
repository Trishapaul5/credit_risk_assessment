import logging
import os
import sys # Import sys for stdout handler
from datetime import datetime

# Define the log file name
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Define the logs directory path
LOGS_DIR = os.path.join(os.getcwd(), "logs")

# CRITICAL FIX 1: Create the directory first
os.makedirs(LOGS_DIR, exist_ok=True)

# Define the full log file path
LOG_FILE_PATH = os.path.join(LOGS_DIR, LOG_FILE)

# CRITICAL FIX 2: Add logging.StreamHandler(sys.stdout) to capture logs in AWS EB
logging.basicConfig(
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler(LOG_FILE_PATH), # Keeps the local file logging
        logging.StreamHandler(sys.stdout)  # Sends logs to standard output for AWS EB
    ]
)

if __name__ == "__main__":
    logging.info("Logging setup test completed, sending logs to file and stdout.")
