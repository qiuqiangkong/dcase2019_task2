import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import argparse
import h5py
import math
import time
import logging
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utilities import (create_folder, get_filename, create_logging, load_scalar, segment_prediction_to_clip_prediction, write_submission)
from data_generator import DataGenerator, TestDataGenerator
from models import (Cnn_5layers_AvgPooling, Cnn_9layers_MaxPooling, 
    Cnn_9layers_AvgPooling, Cnn_13layers_AvgPooling)
from losses import binary_cross_entropy
from evaluate import Evaluator, StatisticsContainer
from pytorch_utils import move_data_to_gpu, forward
import config


def train(args):
    '''Training. Model will be saved after several iterations. 
    
    Args: 
      dataset_dir: string, directory of dataset
      workspace: string, directory of workspace
      train_sources: 'curated' | 'noisy' | 'curated_and_noisy'
      segment_seconds: float, duration of audio recordings to be padded or split
      hop_seconds: float, hop seconds between segments
      pad_type: 'constant' | 'repeat'
      holdout_fold: '1', '2', '3', '4' | 'none', set `none` for training 
          on all data without validation
      model_type: string, e.g. 'Cnn_9layers_AvgPooling'
      batch_size: int
      cuda: bool
      mini_data: bool, set True for debugging on a small part of data
    '''
    
    # Arugments & parameters
    dataset_dir = args.dataset_dir
    workspace = args.workspace
    train_source = args.train_source
    segment_seconds = args.segment_seconds
    hop_seconds = args.hop_seconds
    pad_type = args.pad_type
    holdout_fold = args.holdout_fold
    model_type = args.model_type
    batch_size = args.batch_size
    cuda = args.cuda and torch.cuda.is_available()
    mini_data = args.mini_data
    filename = args.filename
    
    mel_bins = config.mel_bins
    classes_num = config.classes_num
    frames_per_second = config.frames_per_second
    max_iteration = 500      # Number of mini-batches to evaluate on training data
    reduce_lr = False
    
    
    # Paths
    if mini_data:
        prefix = 'minidata_'
    else:
        prefix = ''
                
    curated_feature_hdf5_path = os.path.join(workspace, 'features', 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
        'train_curated.h5')
        
    noisy_feature_hdf5_path = os.path.join(workspace, 'features', 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
        'train_noisy.h5')
        
    curated_cross_validation_path = os.path.join(workspace, 
        'cross_validation_metadata', 'train_curated_cross_validation.csv')

    noisy_cross_validation_path = os.path.join(workspace, 
        'cross_validation_metadata', 'train_noisy_cross_validation.csv')
        
    scalar_path = os.path.join(workspace, 'scalars', 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
        'train_noisy.h5')
        
    checkpoints_dir = os.path.join(workspace, 'checkpoints', filename, 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
        'train_source={}'.format(train_source), 'segment={}s,hop={}s,pad_type={}'.format(
        segment_seconds, hop_seconds, pad_type), 'holdout_fold={}'.format(holdout_fold), 
        model_type)
    create_folder(checkpoints_dir)

    validate_statistics_path = os.path.join(workspace, 'statistics', filename, 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
        'train_source={}'.format(train_source), 'segment={}s,hop={}s,pad_type={}'.format(
        segment_seconds, hop_seconds, pad_type), 'holdout_fold={}'.format(holdout_fold), 
        model_type, 'validate_statistics.pickle')
    create_folder(os.path.dirname(validate_statistics_path))
    
    logs_dir = os.path.join(workspace, 'logs', filename, args.mode, 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
        'train_source={}'.format(train_source), 'segment={}s,hop={}s,pad_type={}'.format(
        segment_seconds, hop_seconds, pad_type), 'holdout_fold={}'.format(holdout_fold), 
        model_type)
    create_logging(logs_dir, 'w')
    logging.info(args)

    # Load scalar
    scalar = load_scalar(scalar_path)
    
    # Model
    Model = eval(model_type)
    model = Model(classes_num)
    
    if cuda:
        model.cuda()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999),
        eps=1e-08, weight_decay=0., amsgrad=True)

    # Data generator
    data_generator = DataGenerator(
        curated_feature_hdf5_path=curated_feature_hdf5_path, 
        noisy_feature_hdf5_path=noisy_feature_hdf5_path, 
        curated_cross_validation_path=curated_cross_validation_path, 
        noisy_cross_validation_path=noisy_cross_validation_path, 
        train_source=train_source, 
        holdout_fold=holdout_fold, 
        segment_seconds=segment_seconds, 
        hop_seconds=hop_seconds, 
        pad_type=pad_type, 
        scalar=scalar, 
        batch_size=batch_size)
    
    # Evaluator
    evaluator = Evaluator(
        model=model, 
        data_generator=data_generator, 
        cuda=cuda)
    
    # Statistics
    validate_statistics_container = StatisticsContainer(validate_statistics_path)
    
    train_bgn_time = time.time()
    iteration = 0
    
    # Train on mini batches
    for batch_data_dict in data_generator.generate_train():
        
        # Evaluate
        if iteration % 500 == 0:
            logging.info('------------------------------------')
            logging.info('Iteration: {}'.format(iteration))

            train_fin_time = time.time()
            
            # Evaluate on partial of train data
            logging.info('Train statistics:')
            
            for target_source in ['curated', 'noisy']:
                validate_curated_statistics = evaluator.evaluate(
                    data_type='train', 
                    target_source=target_source, 
                    max_iteration=max_iteration, 
                    verbose=False)
            
            # Evaluate on holdout validation data
            if holdout_fold != 'none':                
                logging.info('Validate statistics:')
                
                for target_source in ['curated', 'noisy']:
                    validate_curated_statistics = evaluator.evaluate(
                        data_type='validate', 
                        target_source=target_source, 
                        max_iteration=None, 
                        verbose=False)
                        
                    validate_statistics_container.append(
                        iteration, target_source, validate_curated_statistics)
                    
                validate_statistics_container.dump()

            train_time = train_fin_time - train_bgn_time
            validate_time = time.time() - train_fin_time

            logging.info(
                'Train time: {:.3f} s, validate time: {:.3f} s'
                ''.format(train_time, validate_time))

            train_bgn_time = time.time()

        # Save model
        if iteration % 1000 == 0 and iteration > 0:
            checkpoint = {
                'iteration': iteration, 
                'model': model.state_dict(), 
                'optimizer': optimizer.state_dict()}

            checkpoint_path = os.path.join(
                checkpoints_dir, '{}_iterations.pth'.format(iteration))
                
            torch.save(checkpoint, checkpoint_path)
            logging.info('Model saved to {}'.format(checkpoint_path))
            
        # Reduce learning rate
        if reduce_lr and iteration % 200 == 0 and iteration > 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.9
        
        # Move data to GPU
        for key in batch_data_dict.keys():
            if key in ['feature', 'mask', 'target']:
                batch_data_dict[key] = move_data_to_gpu(
                    batch_data_dict[key], cuda)
        
        # Train
        model.train()
        batch_output = model(batch_data_dict['feature'])
        
        # loss
        loss = binary_cross_entropy(batch_output, batch_data_dict['target'])

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Stop learning
        if iteration == 20000:
            break
            
        iteration += 1
        

def inference_validation(args):
    '''Inference and calculate metrics on validation data. 
    
    Args: 
      dataset_dir: string, directory of dataset
      workspace: string, directory of workspace
      train_sources: 'curated' | 'noisy' | 'curated_and_noisy'
      segment_seconds: float, duration of audio recordings to be padded or split
      hop_seconds: float, hop seconds between segments
      pad_type: 'constant' | 'repeat'
      holdout_fold: '1', '2', '3', '4'
      model_type: string, e.g. 'Cnn_9layers_AvgPooling'
      iteration: int, load model of this iteration
      batch_size: int
      cuda: bool
      mini_data: bool, set True for debugging on a small part of data
      visualize: bool, visualize the logmel spectrogram of segments
    '''
    
    # Arugments & parameters
    dataset_dir = args.dataset_dir
    workspace = args.workspace
    train_source = args.train_source
    segment_seconds = args.segment_seconds
    hop_seconds = args.hop_seconds
    pad_type = args.pad_type
    holdout_fold = args.holdout_fold
    model_type = args.model_type
    iteration = args.iteration
    batch_size = args.batch_size
    cuda = args.cuda and torch.cuda.is_available()
    mini_data = args.mini_data
    visualize = args.visualize
    filename = args.filename
    
    mel_bins = config.mel_bins
    classes_num = config.classes_num
    frames_per_second = config.frames_per_second
    
    # Paths
    if mini_data:
        prefix = 'minidata_'
    else:
        prefix = ''
          
    curated_feature_hdf5_path = os.path.join(workspace, 'features', 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
        'train_curated.h5')
        
    noisy_feature_hdf5_path = os.path.join(workspace, 'features', 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
        'train_noisy.h5')
        
    curated_cross_validation_path = os.path.join(workspace, 
        'cross_validation_metadata', 'train_curated_cross_validation.csv')

    noisy_cross_validation_path = os.path.join(workspace, 
        'cross_validation_metadata', 'train_noisy_cross_validation.csv')
        
    scalar_path = os.path.join(workspace, 'scalars', 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
        'train_noisy.h5')
        
    checkpoint_path = os.path.join(workspace, 'checkpoints', filename, 
        'logmel_{}frames_{}melbins'.format(frames_per_second, mel_bins), 
        'train_source={}'.format(train_source), 'segment={}s,hop={}s,pad_type={}'
        ''.format(segment_seconds, hop_seconds, pad_type), 'holdout_fold={}'
        ''.format(holdout_fold), model_type, '{}_iterations.pth'.format(iteration))
        
    figs_dir = os.path.join(workspace, 'figures')
    create_folder(figs_dir)
        
    logs_dir = os.path.join(workspace, 'logs', filename, args.mode, 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
        'train_source={}'.format(train_source), 'segment={}s,hop={}s,pad_type={}'
        ''.format(segment_seconds, hop_seconds, pad_type), 
        'holdout_fold={}'.format(holdout_fold), model_type)
    create_logging(logs_dir, 'w')
    logging.info(args)

    # Load scalar
    scalar = load_scalar(scalar_path)
    
    # Model
    Model = eval(model_type)
    model = Model(classes_num)
    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    
    if cuda:
        model.cuda()
        
    # Data generator
    data_generator = DataGenerator(
        curated_feature_hdf5_path=curated_feature_hdf5_path, 
        noisy_feature_hdf5_path=noisy_feature_hdf5_path, 
        curated_cross_validation_path=curated_cross_validation_path, 
        noisy_cross_validation_path=noisy_cross_validation_path, 
        train_source=train_source, 
        holdout_fold=holdout_fold, 
        segment_seconds=segment_seconds, 
        hop_seconds=hop_seconds, 
        pad_type=pad_type, 
        scalar=scalar, 
        batch_size=batch_size)
    
    # Evaluator
    evaluator = Evaluator(
        model=model, 
        data_generator=data_generator, 
        cuda=cuda)

    # Evaluate
    for target_source in ['curated', 'noisy']:
        validate_curated_statistics = evaluator.evaluate(
            data_type='validate', 
            target_source='curated', 
            max_iteration=None, 
            verbose=True)
        
        # Visualize
        if visualize:
            save_fig_path = os.path.join(figs_dir, 
                '{}_logmel.png'.format(target_source))
            
            validate_curated_statistics = evaluator.visualize(
                data_type='validate', 
                target_source=target_source, 
                save_fig_path=save_fig_path, 
                max_iteration=None, 
                verbose=False)
        
        
def inference_test(args):
    '''Inference and calculate metrics on validation data. 
    
    Args: 
      dataset_dir: string, directory of dataset
      workspace: string, directory of workspace
      train_sources: 'curated' | 'noisy' | 'curated_and_noisy'
      segment_seconds: float, duration of audio recordings to be padded or split
      hop_seconds: float, hop seconds between segments
      pad_type: 'constant' | 'repeat'
      model_type: string, e.g. 'Cnn_9layers_AvgPooling'
      iteration: int, load model of this iteration
      batch_size: int
      cuda: bool
      mini_data: bool, set True for debugging on a small part of data
      visualize: bool, visualize the logmel spectrogram of segments
    '''
    
    # Arugments & parameters
    dataset_dir = args.dataset_dir
    workspace = args.workspace
    train_source = args.train_source
    segment_seconds = args.segment_seconds
    hop_seconds = args.hop_seconds
    pad_type = args.pad_type
    model_type = args.model_type
    iteration = args.iteration
    batch_size = args.batch_size
    cuda = args.cuda and torch.cuda.is_available()
    mini_data = args.mini_data
    filename = args.filename

    holdout_fold = 'none'   # Use model trained on full data without validation
    mel_bins = config.mel_bins
    classes_num = config.classes_num
    frames_per_second = config.frames_per_second
    
    # Paths
    if mini_data:
        prefix = 'minidata_'
    else:
        prefix = ''

    test_feature_hdf5_path = os.path.join(workspace, 'features', 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
        'test.h5')

    scalar_path = os.path.join(workspace, 'scalars', 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
        'train_noisy.h5')
        
    checkpoint_path = os.path.join(workspace, 'checkpoints', filename, 
        'logmel_{}frames_{}melbins'.format(frames_per_second, mel_bins), 
        'train_source={}'.format(train_source), 'segment={}s,hop={}s,pad_type={}'
        ''.format(segment_seconds, hop_seconds, pad_type), 'holdout_fold={}'
        ''.format(holdout_fold), model_type, '{}_iterations.pth'.format(iteration))
        
    submission_path = os.path.join(workspace, 'submissions', filename, 
        'logmel_{}frames_{}melbins'.format(frames_per_second, mel_bins), 
        'train_source={}'.format(train_source), 'segment={}s,hop={}s,pad_type={}'
        ''.format(segment_seconds, hop_seconds, pad_type), 'holdout_fold={}'
        ''.format(holdout_fold), model_type, '{}_iterations_submission.csv'
        ''.format(iteration))
    create_folder(os.path.dirname(submission_path))
        
    # Load scalar
    scalar = load_scalar(scalar_path)
    
    # Model
    Model = eval(model_type)
    model = Model(classes_num)
    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    
    if cuda:
        model.cuda()
        
    # Data generator
    data_generator = TestDataGenerator(
        test_feature_hdf5_path=test_feature_hdf5_path, 
        segment_seconds=segment_seconds, 
        hop_seconds=hop_seconds, 
        pad_type=pad_type, 
        scalar=scalar, 
        batch_size=batch_size)
        
    generate_func = data_generator.generate_test()
    
    # Results of segments
    output_dict = forward(
        model=model, 
        generate_func=generate_func, 
        cuda=cuda)
    
    # Results of audio recordings
    result_dict = segment_prediction_to_clip_prediction(
        output_dict, average='arithmetic')
    
    # Write submission
    write_submission(result_dict, submission_path)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    # Train
    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--dataset_dir', type=str, required=True, help='Directory of dataset.')
    parser_train.add_argument('--workspace', type=str, required=True, help='Directory of your workspace.')
    parser_train.add_argument('--train_source', type=str, choices=['curated', 'noisy', 'curated_and_noisy'], required=True)
    parser_train.add_argument('--segment_seconds', type=float, required=True, help='Segment duration for training.')
    parser_train.add_argument('--hop_seconds', type=float, required=True, help='Hop duration between segments.')
    parser_train.add_argument('--pad_type', type=str, choices=['constant', 'repeat'], required=True, help='Pad short audio recordings with constant silence or repetition.')
    parser_train.add_argument('--holdout_fold', type=str, choices=['1', '2', '3', '4', 'none'], required=True, help='Set `none` for training on all data without validation.')
    parser_train.add_argument('--model_type', type=str, required=True, help='E.g., Cnn_9layers_AvgPooling.')
    parser_train.add_argument('--batch_size', type=int, required=True)
    parser_train.add_argument('--cuda', action='store_true', default=False)
    parser_train.add_argument('--mini_data', action='store_true', default=False, help='Set True for debugging on a small part of data.')

    # Inference validation data
    parser_inference_validation = subparsers.add_parser('inference_validation')
    parser_inference_validation.add_argument('--dataset_dir', type=str, required=True, help='Directory of dataset.')
    parser_inference_validation.add_argument('--workspace', type=str, required=True, help='Directory of your workspace.')
    parser_inference_validation.add_argument('--train_source', type=str, choices=['curated', 'noisy', 'curated_and_noisy'], required=True)
    parser_inference_validation.add_argument('--segment_seconds', type=float, required=True, help='Segment duration for training.')
    parser_inference_validation.add_argument('--hop_seconds', type=float, required=True, help='Hop duration between segments.')
    parser_inference_validation.add_argument('--pad_type', type=str, choices=['constant', 'repeat'], required=True, help='Pad short audio recordings with constant silence or repetition.')
    parser_inference_validation.add_argument('--holdout_fold', type=str, choices=['1', '2', '3', '4', 'none'], required=True, help='Set `none` for training on all data without validation.')
    parser_inference_validation.add_argument('--model_type', type=str, required=True, help='E.g., Cnn_9layers_AvgPooling.')
    parser_inference_validation.add_argument('--iteration', type=int, required=True, help='Load model of this iteration.')
    parser_inference_validation.add_argument('--batch_size', type=int, required=True)
    parser_inference_validation.add_argument('--cuda', action='store_true', default=False)
    parser_inference_validation.add_argument('--visualize', action='store_true', default=False, help='Visualize log mel spectrogram of different sound classes.')
    parser_inference_validation.add_argument('--mini_data', action='store_true', default=False, help='Set True for debugging on a small part of data.')

    # Inference test data
    parser_inference_validation = subparsers.add_parser('inference_test')
    parser_inference_validation.add_argument('--dataset_dir', type=str, required=True, help='Directory of dataset.')
    parser_inference_validation.add_argument('--workspace', type=str, required=True, help='Directory of your workspace.')
    parser_inference_validation.add_argument('--train_source', type=str, choices=['curated', 'noisy', 'curated_and_noisy'], required=True)
    parser_inference_validation.add_argument('--segment_seconds', type=float, required=True, help='Segment duration for training.')
    parser_inference_validation.add_argument('--hop_seconds', type=float, required=True, help='Hop duration between segments.')
    parser_inference_validation.add_argument('--pad_type', type=str, choices=['constant', 'repeat'], required=True, help='Pad short audio recordings with constant silence or repetition.')
    parser_inference_validation.add_argument('--model_type', type=str, required=True, help='E.g., Cnn_9layers_AvgPooling.')
    parser_inference_validation.add_argument('--iteration', type=int, required=True, help='Load model of this iteration.')
    parser_inference_validation.add_argument('--batch_size', type=int, required=True)
    parser_inference_validation.add_argument('--cuda', action='store_true', default=False)
    parser_inference_validation.add_argument('--mini_data', action='store_true', default=False, help='Set True for debugging on a small part of data.')
    
    # Parse arguments
    args = parser.parse_args()
    args.filename = get_filename(__file__)

    if args.mode == 'train':
        train(args)

    elif args.mode == 'inference_validation':
        inference_validation(args)
        
    elif args.mode == 'inference_test':
        inference_test(args)

    else:
        raise Exception('Error argument!')