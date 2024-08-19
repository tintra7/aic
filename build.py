import h5py
import os
import argparse
import faiss
import pandas as pd
import numpy as np
from pathlib import Path
import sys


def peek_features_attributes(h5file):
    with h5py.File(h5file, 'r') as f:
        features_dim = f['data'].shape[1]
        return features_dim
    
def load_features(hdf5_file):
    with h5py.File(hdf5_file, 'r') as h5file:
        ids = h5file['ids'].asstr()[:]
        features = h5file['data'][:]
        ids_and_features = (ids, features)
        return ids_and_features


def create(args):
    # Create new indexing
    print(args.index_file)
    if not args.force and os.path.exists(args.index_file):
        print("Index has exist, skipping creation")
        return
    
    features_files = Path(args.features_dir).glob('*/*.h5')
    features_files = sorted(features_files)
    print(len(features_files))
    ids = []
    feature_vector = []
    for hdf5_file in features_files:
        ids_and_feature = load_features(hdf5_file)
        ids.extend(ids_and_feature[0])
        feature_vector.extend(ids_and_feature[1])
    feature_vector = np.array(feature_vector).astype(np.float32)
    pd.DataFrame({"frame_id": ids}, index=np.arange(0, len(ids))).to_csv("frame_id.csv", index=False) 
    print(feature_vector.shape)
    dim = peek_features_attributes(features_files[0])

    index_type = 'Flat' # Change this

    metric = faiss.METRIC_INNER_PRODUCT
    index = faiss.index_factory(dim, index_type, metric)
    

    if not index.is_trained and hasattr(index, 'train'):
        index.train(feature_vector)
    
    index.add(feature_vector)

    faiss.write_index(index, str(args.index_file))

def main():
    parser = argparse.ArgumentParser(description='FAISS Index Manager: Creates/Updates FAISS indices.')
    parser.add_argument('index_file', type=Path, help='path to the faiss index')
    # parser.add_argument('idmap_file', type=Path, help='path to the id mapping file')

    subparsers = parser.add_subparsers(help='command')

    create_parser = subparsers.add_parser('create', help='Creates a new FAISS index from scratch')
    create_parser.add_argument('--force', default=False, action='store_true', help='overwrite existing data')
    create_parser.add_argument('features_dir', type=Path, help='path to analysis directory containing features h5df files')
    create_parser.set_defaults(func=create)
    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)

if __name__ == "__main__":
    main()
    
    

