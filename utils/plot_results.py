import argparse
import os
import matplotlib.pyplot as plt
import _pickle as cPickle
import numpy as np

import config


def plot_results(args):

    # Arugments & parameters
    workspace = args.workspace
    train_source = args.train_source
    segment_seconds = args.segment_seconds
    hop_seconds = args.hop_seconds
    pad_type = args.pad_type
    mini_data = args.mini_data
    
    filename = 'main'
    frames_per_second = config.frames_per_second
    mel_bins = config.mel_bins
    holdout_fold = 1
    max_plot_iteration = 20000
    iterations = np.arange(0, max_plot_iteration, 500)
    
    # Paths
    if mini_data:
        prefix = 'minidata_'
    else:
        prefix = ''
    
    save_fig_path = 'train_source={}_results.png'.format(train_source)
    
    def _load_stat(model_type, target_source):
        validate_statistics_path = os.path.join(workspace, 'statistics', filename, 
            '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
            'train_source={}'.format(train_source), 'segment={}s,hop={}s,pad_type={}'.format(
            segment_seconds, hop_seconds, pad_type), 'holdout_fold={}'.format(holdout_fold), 
            model_type, 'validate_statistics.pickle')
        
        validate_statistics_dict = cPickle.load(open(validate_statistics_path, 'rb'))
        
        average_precision = np.array([stat['average_precision'] for 
            stat in validate_statistics_dict[target_source]])

        per_class_lwlrap = np.array([stat['per_class_lwlrap'] for 
            stat in validate_statistics_dict[target_source]])

        weight_per_class = np.array([stat['weight_per_class'] for 
            stat in validate_statistics_dict[target_source]])
            
        lwlrap = np.sum(per_class_lwlrap * weight_per_class, axis=-1)
        mAP = np.mean(average_precision, axis=-1)
            
        legend = '{}'.format(model_type)
        
        results = {
            'mAP': mAP, 
            'lwlrap': lwlrap, 
            'legend': legend}
            
        print('Model: {}, target_source: {}'.format(model_type, target_source))
        print('    mAP: {:.3f}'.format(mAP[-1]))
        print('    lwlrap: {:.3f}'.format(lwlrap[-1]))
        
        return results
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    
    for n, target_source in enumerate(['curated', 'noisy']):
        lines = []
        
        results = _load_stat('Cnn_5layers_AvgPooling', target_source=target_source)
        line, = axs[n].plot(results['lwlrap'], label=results['legend'])
        lines.append(line)
        
        results = _load_stat('Cnn_9layers_AvgPooling', target_source=target_source)
        line, = axs[n].plot(results['lwlrap'], label=results['legend'])
        lines.append(line)
        
        results = _load_stat('Cnn_9layers_MaxPooling', target_source=target_source)
        line, = axs[n].plot(results['lwlrap'], label=results['legend'])
        lines.append(line)
        
        results = _load_stat('Cnn_13layers_AvgPooling', target_source=target_source)
        line, = axs[n].plot(results['lwlrap'], label=results['legend'])
        lines.append(line)
    
    axs[0].set_title('Target source: {}'.format('curated'))
    axs[1].set_title('Target source: {}'.format('noisy'))
    
    for i in range(2):
        axs[i].legend(handles=lines, loc=4)
        axs[i].set_ylim(0, 1.0)
        axs[i].set_xlabel('Iterations')
        axs[i].set_ylabel('lwlrap')
        axs[i].grid(color='b', linestyle='solid', linewidth=0.2)
        axs[i].xaxis.set_ticks(np.arange(0, len(iterations) + 1, len(iterations) // 4))
        axs[i].xaxis.set_ticklabels(np.arange(0, max_plot_iteration + 1, max_plot_iteration // 4))
    
    plt.tight_layout()
    plt.savefig(save_fig_path)
    print('Figure saved to {}'.format(save_fig_path))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--workspace', type=str, required=True, help='Directory of your workspace.')
    parser.add_argument('--train_source', type=str, choices=['curated', 'noisy', 'curated_and_noisy'], required=True)
    parser.add_argument('--segment_seconds', type=float, required=True, help='Segment duration for training.')
    parser.add_argument('--hop_seconds', type=float, required=True, help='Hop duration between segments.')
    parser.add_argument('--pad_type', type=str, choices=['constant', 'repeat'], required=True, help='Pad short audio recordings with constant silence or repetition.')
    parser.add_argument('--mini_data', action='store_true', default=False, help='Set True for debugging on a small part of data.')

    args = parser.parse_args()
    
    plot_results(args)