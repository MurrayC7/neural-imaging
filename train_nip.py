#!/usr/bin/env python3
# coding: utf-8
import os
import sys
import json
import argparse

import numpy as np

from helpers import coreutils, dataset
from training.pipeline import train_nip_model

# Set progress bar width
TQDM_WIDTH = 120

# Disable unimportant logging and import TF
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def main():
    parser = argparse.ArgumentParser(description='Train a neural imaging pipeline')
    parser.add_argument('--cam', dest='camera', action='store', help='camera')
    parser.add_argument('--nip', dest='nips', action='append', choices=["INet", "UNet", "DNet", "OctUNet", "UNet3D"],
                        help='add NIP for training (repeat if needed)')
    parser.add_argument('--out', dest='out_dir', action='store', default='./checkpoint/nip_model_snapshots',
                        help='output directory for storing trained NIP models')
    parser.add_argument('--data', dest='data_dir', action='store', default='../../datasets/raw/nip_training_data/',
                        help='input directory with training data (.npy and .png pairs)')
    parser.add_argument('--patch', dest='patch_size', action='store', default=512, type=int,
                        help='training patch size (RGB)')
    parser.add_argument('--epochs', dest='epochs', action='store', default=25000, type=int,
                        help='maximum number of training epochs')
    parser.add_argument('--batch', dest='batch_size', action='store', default=20, type=int,
                        help='training batch size')
    parser.add_argument('--params', dest='nip_params', default=None,
                        help='Extra parameters for NIP constructor (JSON string)')
    parser.add_argument('--resume', dest='resume', action='store_true', default=False,
                        help='Resume training from last checkpoint, if possible')
    parser.add_argument('--split', dest='split', action='store', default='270:30:1',
                        help='data split with #training:#validation:#validation_patches - e.g., 120:30:1')
    parser.add_argument('--ext', dest='extension', action='store', default='png',
                        help='file extension of rgb images - e.g., png, JPG')

    args = parser.parse_args()

    if not args.camera:
        print('A camera needs to be specified!')
        parser.print_usage()
        sys.exit(1)

    if not args.nips:
        print('At least one NIP needs to be specified!')
        parser.print_usage()
        sys.exit(1)

    data_directory = os.path.join(args.data_dir, args.camera)
    out_directory_root = args.out_dir

    try:
        if args.nip_params is not None:
            args.nip_params = json.loads(args.nip_params.replace('\'', '"'))
    except json.decoder.JSONDecodeError:
        print('WARNING', 'JSON parsing error for: ', args.nip_params.replace('\'', '"'))
        sys.exit(2)

    print('## Parameters summary')
    print('Camera : {}'.format(args.camera))
    print('NIPs   : {}'.format(args.nips))
    print('Params : {}'.format(args.nip_params))
    print('Input  : {}'.format(data_directory))
    print('Output : {}'.format(out_directory_root))
    print('Resume : {}'.format(args.resume))

    # Load training and validation data
    training_spec = {
        'seed': 1234,
        'n_images': int(args.split.split(':')[0]),
        'v_images': int(args.split.split(':')[1]),
        'valid_patches': int(args.split.split(':')[2]),
        'valid_patch_size': 512,
    }

    np.random.seed(training_spec['seed'])

    # Load and summarize the training data
    data = dataset.IPDataset(data_directory, n_images=training_spec['n_images'], v_images=training_spec['v_images'],
                             load='xy', val_rgb_patch_size=training_spec['valid_patch_size'],
                             val_n_patches=training_spec['valid_patches'], rgb_extension=args.extension)

    for key in ['Training', 'Validation']:
        print('{:>16s} [{:5.1f} GB] : X -> {}, Y -> {} '.format(
            '{} data'.format(key),
            coreutils.mem(data[key.lower()]['x']) + coreutils.mem(data[key.lower()]['y']),
            data[key.lower()]['x'].shape,
            data[key.lower()]['y'].shape
        ), flush=True)

    # Lazy loading to prevent delays in basic CLI interaction
    from models import pipelines
    import tensorflow as tf

    # Train the Desired NIP Models
    for pipe in args.nips:

        if not issubclass(getattr(pipelines, pipe), pipelines.NIPModel):
            supported_nips = [x for x in dir(pipelines) if
                              x != 'NIPModel' and type(getattr(pipelines, x)) is type and issubclass(
                                  getattr(pipelines, x), pipelines.NIPModel)]
            raise ValueError('Invalid NIP model ({})! Available NIPs: ({})'.format(pipe, supported_nips))

        args.nip_params = args.nip_params or {}

        tf.reset_default_graph()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
        config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        model = getattr(pipelines, pipe)(sess, tf.get_default_graph(), loss_metric='L1', **args.nip_params)
        model.sess.run(tf.global_variables_initializer())

        train_nip_model(model, args.camera, args.epochs, validation_loss_threshold=1e-5, patch_size=args.patch_size,
                        resume=args.resume, sampling_rate=1000, batch_size=args.batch_size, learning_rate=1e-4,
                        data=data,
                        out_directory_root=args.out_dir)

        sess.close()

    return


if __name__ == "__main__":
    main()
