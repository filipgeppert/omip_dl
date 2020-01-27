import os
import sys
from src.plan_classifier import run_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, 'pytorch_otodom'))

if __name__ == "__main__":
    print(BASE_DIR)
    run_model()
