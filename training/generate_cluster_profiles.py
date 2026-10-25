import os
import sys

# Ensure root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from training.train_clustering import train_clustering

def main():
    print("Generating vehicle cluster profiles and saving centroids...")
    train_clustering(optimal_k=3)
    print("Cluster profiles generation complete.")

if __name__ == '__main__':
    main()
