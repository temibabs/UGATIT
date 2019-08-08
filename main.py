import argparse

import tensorflow as tf

from model import UGATIT
from utils import *


def parse_args() -> argparse.Namespace:
    desc = 'Tensorflow implementation of U-GAT-IT'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--phase', type=str, default='train', help='[train / test]')
    parser.add_argument('--light', type=str2bool, default=False, help='[U-GAT-IT full version / U-GAT-IT light version')
    parser.add_argument('--dataset', type=str, default='selfie2anime', help='dataset_name')

    parser.add_argument('--epoch', type=int, default=100, help='The number of epochs to run')
    parser.add_argument('--iteration', type=int, default=10000, help='The number of training iterations')
    parser.add_argument('--batch_size', type=int, default=1, help='The size of batches')
    parser.add_argument('--print_freq', type=int, default=1000, help='The frequency of print')
    parser.add_argument('--decay_flag', type=str2bool, default=True, help='The decay flag')
    parser.add_argument('--decay_epoch', type=int, default=50, help='Decay epoch')

    parser.add_argument('--lr', type=float, default=0.0001, help='The learning rate')
    parser.add_argument('--GP_ld', type=int, default=10, help='The gradient penalty lambda')
    parser.add_argument('--adv_weight', type=int, default=1, help='Weight about GAN')
    parser.add_argument('--cycle_weight', type=int, default=10, help='Weight about Cycle')
    parser.add_argument('--identity_weight', type=int, default=10, help='Weight about Identity')
    parser.add_argument('--cam_weight', type=int, default=1000, help='Weight about CAM')
    parser.add_argument('--gan_type', type=str, default='lsgan', help='[gan / lsgan / wgan-up / dragan / hinge]')

    parser.add_argument('--smoothing', type=str2bool, default=False, help='AdaLIN smoothing effect')

    parser.add_argument('--ch', type=int, default=64, help='Base channel number per layer')
    parser.add_argument('--n_res', type=int, default=4, help='The number of resblocks')
    parser.add_argument('--n_dis', type=int, default=6, help='The number of discriminator layers')
    parser.add_argument('--n_critic', type=int, default=1, help='The number of critic')
    parser.add_argument('--sn', type=str2bool, default=True, help='Whether to use Spectral Norm')

    parser.add_argument('--image_size', type=int, default=256, help='The size of the image')
    parser.add_argument('--image_ch', type=int, default=3, help='The size of image channel')
    parser.add_argument('--augment_flag', type=str2bool, default=True, help='Whether to perform Image augmentation')

    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint', help='Directoty for checkpoints')
    parser.add_argument('--result_dir', type=str, default='results', help='Directory to save generated images')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory to save training logs')
    parser.add_argument('--sample_dir', type=str, default='samples', help='Directory name to save the samples on training')

    return check_args(parser.parse_args())


def check_args(arguments: argparse.Namespace) -> argparse.Namespace:
    check_folder(arguments.checkpoint_dir)
    check_folder(arguments.result_dir)
    check_folder(arguments.log_dir)
    check_folder(arguments.sample_dir)

    try:
        assert arguments.epoch >= 1
    except ValueError:
        print('Number of epochs must be >= 1')

    try:
        assert arguments.batch_size >= 1
    except ValueError:
        print('Batch size must be >= 1')

    return arguments


def main():
    arguments = parse_args()
    if arguments is None:
        exit()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        model = UGATIT(sess, args=arguments)
        model.build_model()
        
        show_all_variables()
        
        if arguments.phase == 'train':
            model.train()
            print('[*] Training finished')
            
        if arguments.phase == 'test':
            model.test()
            print('[+] Test finished')

if __name__ == '__main__':
    main()
