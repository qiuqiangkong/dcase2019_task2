import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))

import numpy as np
import time
import logging
import matplotlib.pyplot as plt
from sklearn import metrics
import datetime
import _pickle as cPickle
import sed_eval

from utilities import (get_filename, inverse_scale, 
    segment_prediction_to_clip_prediction)
from pytorch_utils import forward
from lwlrap import calculate_per_class_lwlrap
import config


class Evaluator(object):
    def __init__(self, model, data_generator, cuda=True):
        '''Evaluator to evaluate prediction performance. 
        
        Args: 
          model: object
          data_generator: object
          cuda: bool
        '''
        
        self.model = model
        self.data_generator = data_generator
        self.cuda = cuda
        
        self.frames_per_second = config.frames_per_second
        self.labels = config.labels
        self.idx_to_lb = config.idx_to_lb
        self.classes_num = config.classes_num

    def evaluate(self, data_type, target_source, max_iteration=None, 
        verbose=False):
        '''Evaluate the performance. 
        
        Args: 
          data_type: 'train' | 'validate'
          target_source: 'curated' | 'noisy'
          max_iteration: None | int, maximum iteration to run to speed up evaluation
          verbose: bool
        '''
        
        assert(data_type in ['train', 'validate'])
        assert(target_source in ['curated', 'noisy'])
        
        generate_func = self.data_generator.generate_validate(
            data_type=data_type, 
            target_source=target_source, 
            max_iteration=max_iteration)
        
        # Results of segments
        output_dict = forward(
            model=self.model, 
            generate_func=generate_func, 
            cuda=self.cuda, 
            return_target=True)
        
        # Results of audio recordings
        result_dict = segment_prediction_to_clip_prediction(
            output_dict, average='arithmetic')
        
        output = result_dict['output']
        target = result_dict['target']
        
        # Mean average precision
        average_precision = metrics.average_precision_score(
            target, output, average=None)
        mAP = np.mean(average_precision)
        
        # Label-weighted label-ranking average precision
        (per_class_lwlrap, weight_per_class) = calculate_per_class_lwlrap(
            target, output)            
        mean_lwlrap = np.sum(per_class_lwlrap * weight_per_class)
        
        logging.info('    Target source: {}, mAP: {:.3f}, mean_lwlrap: {:.3f}'
            ''.format(target_source, mAP, mean_lwlrap))
        
        statistics = {
            'average_precision': average_precision, 
            'per_class_lwlrap': per_class_lwlrap, 
            'weight_per_class': weight_per_class}
            
        if verbose:
            for n in range(self.classes_num):
                logging.info('    {:<20}{:.3f}'.format(self.labels[n], 
                    per_class_lwlrap[n]))
            logging.info('')
        
        return statistics
            
    def visualize(self, data_type, target_source, save_fig_path, 
        max_iteration=None):
        '''Visualize logmel of different sound classes. 
        
        Args: 
          data_type: 'train' | 'validate'
          target_source: 'curated' | 'noisy'
          save_fig_path: string, path to save figure
          max_iteration: None | int, maximum iteration to run to speed up evaluation
        '''

        generate_func = self.data_generator.generate_validate(
            data_type=data_type, 
            target_source=target_source, 
            max_iteration=max_iteration)
        
        # Results of segments
        output_dict = forward(
            model=self.model, 
            generate_func=generate_func, 
            cuda=self.cuda, 
            return_target=True, 
            return_input=True)

        target = output_dict['target']
        output = output_dict['output']
        feature = output_dict['feature']
        
        (audios_num, segment_frames, mel_bins) = feature.shape
        segment_duration = segment_frames / self.frames_per_second

        # Plot log mel spectrogram of different sound classes
        rows_num = 10
        cols_num = 8
        
        fig, axs = plt.subplots(rows_num, cols_num, figsize=(15, 15))

        for k in range(self.classes_num):
            for n, audio_name in enumerate(output_dict['audio_name']):
                if target[n, k] == 1:
                    title = self.idx_to_lb[k][0:20]
                    row = k // cols_num
                    col = k % cols_num
                    axs[row, col].set_title(title, color='r', fontsize=9)
                    logmel = inverse_scale(feature[n], 
                        self.data_generator.scalar['mean'], 
                        self.data_generator.scalar['std'])
                    axs[row, col].matshow(logmel.T, origin='lower', 
                        aspect='auto', cmap='jet')
                    axs[row, col].set_xticks([0, segment_frames])
                    axs[row, col].set_xticklabels(
                        ['0', '{:.1f} s'.format(segment_duration)], fontsize=6)
                    axs[row, col].xaxis.set_ticks_position('bottom')
                    axs[row, col].set_ylabel('Mel bins', fontsize=7)
                    axs[row, col].set_yticks([])
                    break
        
        for k in range(self.classes_num, rows_num * cols_num):
            row = k // cols_num
            col = k % cols_num
            axs[row, col].set_visible(False)
            
        plt.tight_layout(pad=0, w_pad=0, h_pad=0)
        plt.savefig(save_fig_path)
        logging.info('Save figure to {}'.format(save_fig_path))


class StatisticsContainer(object):
    def __init__(self, statistics_path):
        '''Container of statistics during training. 
        
        Args:
          statistics_path: string, path to write out
        '''
        self.statistics_path = statistics_path

        self.backup_statistics_path = '{}_{}.pickle'.format(
            os.path.splitext(self.statistics_path)[0], 
                datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

        self.statistics_dict = {'curated': [], 'noisy': []}

    def append(self, iteration, target_source, statistics):
        statistics['iteration'] = iteration
        self.statistics_dict[target_source].append(statistics)

    def dump(self):
        cPickle.dump(self.statistics_dict, open(self.statistics_path, 'wb'))
        cPickle.dump(self.statistics_dict, open(self.backup_statistics_path, 'wb'))
        logging.info('    Dump statistics to {}'.format(self.statistics_path))