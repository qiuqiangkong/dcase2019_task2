import os
import sys
import numpy as np
import soundfile
import librosa
import h5py
import math
import pandas as pd
from sklearn import metrics
import logging
import matplotlib.pyplot as plt

import config


def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)


def get_filename(path):
    path = os.path.realpath(path)
    name_ext = path.split('/')[-1]
    name = os.path.splitext(name_ext)[0]
    return name


def create_logging(log_dir, filemode):
    create_folder(log_dir)
    i1 = 0

    while os.path.isfile(os.path.join(log_dir, '{:04d}.log'.format(i1))):
        i1 += 1
        
    log_path = os.path.join(log_dir, '{:04d}.log'.format(i1))
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
        datefmt='%a, %d %b %Y %H:%M:%S',
        filename=log_path,
        filemode=filemode)

    # Print to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    
    return logging
    
    
def read_audio(audio_path, target_fs=None):
    (audio, fs) = soundfile.read(audio_path)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
        
    if target_fs is not None and fs != target_fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
        fs = target_fs
        
    return audio, fs
    
    
def calculate_scalar_of_tensor(x):
    if x.ndim == 2:
        axis = 0
    elif x.ndim == 3:
        axis = (0, 1)

    mean = np.mean(x, axis=axis)
    std = np.std(x, axis=axis)

    return mean, std


def load_scalar(scalar_path):
    with h5py.File(scalar_path, 'r') as hf:
        mean = hf['mean'][:]
        std = hf['std'][:]
        
    scalar = {'mean': mean, 'std': std}
    return scalar
    
    
def scale(x, mean, std):
    return (x - mean) / std
    
    
def inverse_scale(x, mean, std):
    return x * std + mean
    
    
def read_metadata(metadata_path, lb_to_idx):
    '''Read metadata from a csv file. 
    
    Returns:
      meta_dict: {'audio_name': (audios_num,), 
                  'target': (audios_num, classes_num)}
    '''
    df = pd.read_csv(metadata_path, sep=',')
    
    audios_num = len(df)
    classes_num = len(lb_to_idx)
    
    target = np.zeros((audios_num, classes_num), dtype=np.bool)
    
    for n, labels in enumerate(df['labels']):
        for label in labels.split(','):
            idx = lb_to_idx[label]
            target[n, idx] = 1
            
    meta_dict = {}
    meta_dict['audio_name'] = np.array(df['fname'].tolist())
    meta_dict['target'] = target

    return meta_dict
    
    
def segment_prediction_to_clip_prediction(output_dict, average):
    '''Aggregate segment-level predictions to clip-level prediction. 
    
    Args:
      output_dict: {'audio_name': (segments_num,), 
                    'output': (segments_num, classes_num), 
                    (if exist) 'target': (segments_num,)}
      average: 'arithmetic' | 'geometric'
      
    Returns:
      result_dict: {'audio_name': (audios_num,), 
                    'output': (audios_num, classes_num), 
                    'target': (audios_num, classes_num)}
    '''
    
    assert average in ['arithmetic', 'geometric']
    
    audio_names = np.array(sorted(set(output_dict['audio_name'])))
    outputs = []    
    segment_output_dict = {}
    
    has_target = 'target' in output_dict.keys()
    
    if has_target:
        targets = []
    
    '''segment_output_dict: {'a.wav': [(classes_num,), ...]
                             'b.wav': [(classes_num,), (classes_num,), ...]
                             ...}'''
    for n, audio_name in enumerate(output_dict['audio_name']):
        if audio_name not in segment_output_dict.keys():
            segment_output_dict[audio_name] = [output_dict['output'][n]]
            if has_target:
                targets.append(output_dict['target'][n])
        else:
            segment_output_dict[audio_name].append(output_dict['output'][n])
            
    # Aggregate results in segment_output_dict
    for audio_name in audio_names:
        segment_outputs = np.array(segment_output_dict[audio_name])
        if average == 'arithmetic':
            output = np.mean(segment_outputs, axis=0)
        elif average == 'geometric':
            output = np.exp(np.mean(np.log(segment_outputs), axis=0))
        outputs.append(output)
    
    outputs = np.array(outputs)
    
    result_dict = {
        'audio_name': audio_names, 
        'output': outputs}
        
    if has_target:
        targets = np.array(targets)
        result_dict['target'] = targets
    
    return result_dict
    
    
def write_submission(result_dict, submission_path):
    
    labels = config.labels
    
    fw = open(submission_path, 'w')
    
    fw.write('fname')
    for label in labels:
        fw.write(',{}'.format(label))
    fw.write('\n')
    
    for n, audio_name in enumerate(result_dict['audio_name']):
        fw.write(audio_name)
        for prob in result_dict['output'][n]:
            fw.write(',{}'.format(prob))
        fw.write('\n')
    fw.close()
    print('Write submission to {}'.format(submission_path))