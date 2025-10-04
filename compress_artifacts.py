import os
import sys
import joblib

# Import custom project utilities (assuming PYTHONPATH is set up correctly)
try:
    from src.utils import load_object
    from src.logger import logging
except ImportError:
    print("FATAL: Could not import project utilities. Ensure src/ is a package and PYTHONPATH is set.")
    sys.exit(1)

def compress_artifacts(model_path, preprocessor_path):
    """
    Loads model and preprocessor artifacts and saves them back with GZIP compression.
    """
    if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
        logging.error("Artifact files not found. Run training_pipeline.py first.")
        return

    try:
        logging.info("Starting artifact compression using joblib...")
        
        # 1. Load artifacts first (using the load_object defined in src/utils.py)
        model = load_object(model_path)
        preprocessor = load_object(preprocessor_path)

        # 2. Save with GZIP compression (level 3 is a good balance of speed/ratio)
        joblib.dump(model, model_path, compress=('gzip', 3))
        joblib.dump(preprocessor, preprocessor_path, compress=('gzip', 3))
        
        logging.info("Compression complete. Artifacts overwritten with GZIP compression (level 3).")
        
    except Exception as e:
        logging.error(f"Error during artifact compression: {e}")
        sys.exit(1)

if __name__ == "__main__":
    MODEL_PATH = 'artifacts/model.pkl'
    PREPROCESSOR_PATH = 'artifacts/preprocessor.pkl'
    
    # Execute compression
    compress_artifacts(MODEL_PATH, PREPROCESSOR_PATH)