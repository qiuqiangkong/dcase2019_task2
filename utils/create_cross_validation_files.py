import argparse
import os
import pandas as pd
import numpy as np

from utilities import create_folder
import config


def create_cross_validation_file(args):
    '''Create and write out cross validation file. 
    
    Args:
      dataset_dir: string, directory of dataset
      workspace: string, directory of workspace
      data_type: 'train_curated' | 'train_noisy'
    '''
    
    # Arugments & parameters
    dataset_dir = args.dataset_dir
    workspace = args.workspace
    data_type = args.data_type
    folds_num = config.folds_num
    
    # Paths
    metadata_path = os.path.join(dataset_dir, '{}.csv'.format(data_type))
    
    cross_validation_path = os.path.join(workspace, 'cross_validation_metadata', 
        '{}_cross_validation.csv'.format(data_type))
    create_folder(os.path.dirname(cross_validation_path))
    
    # Read meta data
    df = pd.read_csv(metadata_path, sep=',')
    
    # Create cross validation file
    new_df = pd.DataFrame()
    new_df['fname'] = df['fname']
    new_df['fold'] = np.arange(len(df)) % folds_num + 1
    
    new_df.to_csv(cross_validation_path)
    
    print('Write cross validation file to {}'.format(cross_validation_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create cross validation files. ')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Directory of dataset.')    
    parser.add_argument('--workspace', type=str, required=True, help='Directory of your workspace.')        
    parser.add_argument('--data_type', type=str, choices=['train_curated', 'train_noisy'], required=True)        
    
    # Parse arguments
    args = parser.parse_args()
    
    create_cross_validation_file(args)