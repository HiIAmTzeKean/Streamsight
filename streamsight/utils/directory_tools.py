import os

def safe_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)