import sys
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perception 3D main program')
    parser.add_argument('--phase', required=True, choices=['train', 'val', 'test'], default='train')
    args = parser.parse_args()
    