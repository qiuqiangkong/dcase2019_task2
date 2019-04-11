import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], 'utils'))
import numpy as np
import argparse
import h5py
import librosa
from scipy import signal
import matplotlib.pyplot as plt
import time
import math
import pandas as pd
import random

from utilities import (create_folder, read_audio, calculate_scalar_of_tensor, 
    read_metadata)
import config


class LogMelExtractor(object):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax):
        '''Log mel feature extractor. 
        
        Args:
          sample_rate: int
          window_size: int
          hop_size: int
          mel_bins: int
          fmin: int, minimum frequency of mel filter banks
          fmax: int, maximum frequency of mel filter banks
        '''
        
        self.window_size = window_size
        self.hop_size = hop_size
        self.window_func = np.hanning(window_size)
        
        self.melW = librosa.filters.mel(
            sr=sample_rate, 
            n_fft=window_size, 
            n_mels=mel_bins, 
            fmin=fmin, 
            fmax=fmax).T
        '''(n_fft // 2 + 1, mel_bins)'''

    def transform(self, audio):
        '''Extract feature of a singlechannel audio file. 
        
        Args:
          audio: (samples,)
          
        Returns:
          feature: (frames_num, freq_bins)
        '''
    
        window_size = self.window_size
        hop_size = self.hop_size
        window_func = self.window_func
        
        # Compute short-time Fourier transform
        stft_matrix = librosa.core.stft(
            y=audio, 
            n_fft=window_size, 
            hop_length=hop_size, 
            window=window_func, 
            center=True, 
            dtype=np.complex64, 
            pad_mode='reflect').T
        '''(N, n_fft // 2 + 1)'''
    
        # Mel spectrogram
        mel_spectrogram = np.dot(np.abs(stft_matrix) ** 2, self.melW)
        
        # Log mel spectrogram
        logmel_spectrogram = librosa.core.power_to_db(
            mel_spectrogram, ref=1.0, amin=1e-10, 
            top_db=None)
        
        logmel_spectrogram = logmel_spectrogram.astype(np.float32)
        
        return logmel_spectrogram


def calculate_feature_for_all_audio_files(args):
    '''Calculate feature of audio files and write out features to a hdf5 file. 
    
    Args:
      dataset_dir: string
      workspace: string
      data_type: 'train_curated', 'train_noisy', 'test'
      mini_data: bool, set True for debugging on a small part of data
    '''

    # Arguments & parameters
    dataset_dir = args.dataset_dir
    workspace = args.workspace
    data_type = args.data_type
    mini_data = args.mini_data
    
    sample_rate = config.sample_rate
    window_size = config.window_size
    hop_size = config.hop_size
    mel_bins = config.mel_bins
    fmin = config.fmin
    fmax = config.fmax
    frames_per_second = config.frames_per_second
    lb_to_idx = config.lb_to_idx
    
    # Paths
    if mini_data:
        prefix = 'minidata_'
    else:
        prefix = ''
    
    if data_type in ['train_curated', 'train_noisy']:
        metadata_path = os.path.join(dataset_dir, '{}.csv'.format(data_type))
    else:
        pass
        
    audios_dir = os.path.join(dataset_dir, data_type)
    
    feature_path = os.path.join(workspace, 'features', 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
        '{}.h5'.format(data_type))
    create_folder(os.path.dirname(feature_path))
    
    # Read meta data
    if data_type in ['train_curated', 'train_noisy']:
        meta_dict = read_metadata(metadata_path, lb_to_idx)
    elif data_type == 'test':
        meta_dict = {'audio_name': np.array(sorted(os.listdir(audios_dir)))}
        
    # Feature extractor
    feature_extractor = LogMelExtractor(
        sample_rate=sample_rate, 
        window_size=window_size, 
        hop_size=hop_size, 
        mel_bins=mel_bins, 
        fmin=fmin, 
        fmax=fmax)

    if mini_data:
        mini_num = 100
        total_num = len(meta_dict['audio_name'])
        random_state = np.random.RandomState(1234)
        indexes = random_state.choice(total_num, size=mini_num, replace=False)
        meta_dict['audio_name'] = meta_dict['audio_name'][indexes]
        if 'target' in meta_dict:
            meta_dict['target'] = meta_dict['target'][indexes]
    
    # Hdf5 file for storing features and targets
    print('Extracting features of all audio files ...')
    extract_time = time.time()
    
    audios_num = len(meta_dict['audio_name'])
    
    hf = h5py.File(feature_path, 'w')

    hf.create_dataset(
        name='audio_name', 
        data=[audio_name.encode() for audio_name in meta_dict['audio_name']], 
        dtype='S20')
    
    if 'target' in meta_dict:
        hf.create_dataset(
            name='target', 
            data=meta_dict['target'], 
            dtype=np.bool)
        
    hf.create_dataset(
        name='feature', 
        shape=(0, mel_bins), 
        maxshape=(None, mel_bins), 
        dtype=np.float32)
        
    hf.create_dataset(
        name='begin_index', 
        shape=(audios_num,),
        dtype=np.int32)
        
    hf.create_dataset(
        name='end_index', 
        shape=(audios_num,),
        dtype=np.int32)
        
    for (n, audio_name) in enumerate(meta_dict['audio_name']):
        audio_path = os.path.join(audios_dir, audio_name)
        print(n, audio_path)
        
        # Read audio
        (audio, _) = read_audio(
            audio_path=audio_path, 
            target_fs=sample_rate)
    
        # Extract feature
        feature = feature_extractor.transform(audio)
        print(feature.shape)
        
        begin_index = hf['feature'].shape[0]
        end_index = begin_index + feature.shape[0]
        hf['feature'].resize((end_index, mel_bins))
        hf['feature'][begin_index : end_index, :] = feature
        
        hf['begin_index'][n] = begin_index
        hf['end_index'][n] = end_index

    hf.close()
        
    print('Write hdf5 file to {} using {:.3f} s'.format(
        feature_path, time.time() - extract_time))
    
    
def calculate_scalar(args):
    '''Calculate and write out scalar of features. 
    
    Args:
      workspace: string
      data_type: 'train_curated'
      mini_data: bool, set True for debugging on a small part of data
    '''

    # Arguments & parameters
    workspace = args.workspace
    data_type = args.data_type
    mini_data = args.mini_data
    
    mel_bins = config.mel_bins
    frames_per_second = config.frames_per_second
    
    # Paths
    if mini_data:
        prefix = 'minidata_'
    else:
        prefix = ''
    
    feature_path = os.path.join(workspace, 'features', 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
        '{}.h5'.format(data_type))
        
    scalar_path = os.path.join(workspace, 'scalars', 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
        '{}.h5'.format(data_type))
    create_folder(os.path.dirname(scalar_path))
        
    # Load data
    load_time = time.time()
    
    with h5py.File(feature_path, 'r') as hf:
        features = hf['feature'][:]
    
    # Calculate scalar
    (mean, std) = calculate_scalar_of_tensor(features)
    
    with h5py.File(scalar_path, 'w') as hf:
        hf.create_dataset('mean', data=mean, dtype=np.float32)
        hf.create_dataset('std', data=std, dtype=np.float32)
    
    print('All features: {}'.format(features.shape))
    print('mean: {}'.format(mean))
    print('std: {}'.format(std))
    print('Write out scalar to {}'.format(scalar_path))
            

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='')
    subparsers = parser.add_subparsers(dest='mode')

    # Calculate feature for all audio files
    parser_logmel = subparsers.add_parser('calculate_feature_for_all_audio_files')    
    parser_logmel.add_argument('--dataset_dir', type=str, required=True, help='Directory of dataset.')    
    parser_logmel.add_argument('--workspace', type=str, required=True, help='Directory of your workspace.')        
    parser_logmel.add_argument('--data_type', type=str, choices=['train_curated', 'train_noisy', 'test'], required=True)        
    parser_logmel.add_argument('--mini_data', action='store_true', default=False, help='Set True for debugging on a small part of data.')
        
    # Calculate scalar
    parser_scalar = subparsers.add_parser('calculate_scalar')    
    parser_scalar.add_argument('--workspace', type=str, required=True, help='Directory of your workspace.')    
    parser_scalar.add_argument('--data_type', type=str, choices=['train_noisy'], required=True, help='Scalar is calculated on train_noisy data. ')        
    parser_scalar.add_argument('--mini_data', action='store_true', default=False, help='Set True for debugging on a small part of data.')
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.mode == 'calculate_feature_for_all_audio_files':
        calculate_feature_for_all_audio_files(args)
        
    elif args.mode == 'calculate_scalar':
        calculate_scalar(args)
        
    else:
        raise Exception('Incorrect arguments!')