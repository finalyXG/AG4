import argparse
import os
import scipy.misc
import numpy as np
import tensorflow as tf
from model_lbxx import cyclegan
import crash_on_ipy 
import ipdb
import imageio

##
parser = argparse.ArgumentParser(description='')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=4, help='# images in batch')
parser.add_argument('--dataset_dir', dest='dataset_dir', default='lbxx_02', help='path of the dataset')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint_lbxx', help='models are saved here')
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample_lbxx', help='sample are saved here')

#parser.add_argument('--dataset_dir', dest='dataset_dir', default='horse2zebra_wgan', help='path of the dataset')
parser.add_argument('--epoch', dest='epoch', type=int, default=200, help='# of epoch')
parser.add_argument('--train_size', dest='train_size', type=int, default=1e8, help='# images used to train')
parser.add_argument('--load_size', dest='load_size', type=int, default=128, help='scale images to this size')
parser.add_argument('--fine_size', dest='fine_size', type=int, default=128, help='then crop to this size')

parser.add_argument('--frame_os', dest='frame_os', type=int, default=128, help='video wdith with croping')
parser.add_argument('--frame_w', dest='frame_w', type=int, default=128, help='video wdith with croping')
parser.add_argument('--frame_h', dest='frame_h', type=int, default=128, help='video height with croping')
parser.add_argument('--frame_cs', dest='frame_cs', type=int, default=96, help='video with croping')

parser.add_argument('--frames_nb', dest='frames_nb', type=int, default=8, help='number of anime frames to be predicted')

parser.add_argument('--z_dim', dest='z_dim', type=int, default=100, help='# the dimension of z')
parser.add_argument('--ngf', dest='ngf', type=int, default=32, help='# of gen filters in first conv layer')
parser.add_argument('--ndf', dest='ndf', type=int, default=64, help='# of discri filters in first conv layer')
parser.add_argument('--input_nc', dest='input_nc', type=int, default=3, help='# of input image channels')
parser.add_argument('--input_ic', dest='input_ic', type=int, default=9, help='# of input channels')

parser.add_argument('--output_nc', dest='output_nc', type=int, default=3, help='# of output image channels')
parser.add_argument('--niter', dest='niter', type=int, default=200, help='# of iter at starting learning rate')
parser.add_argument('--lr', dest='lr', type=float, default=2e-4, help='initial learning rate for adam')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--flip', dest='flip', type=bool, default=True, help='if flip the images for data argumentation')
parser.add_argument('--which_direction', dest='which_direction', default='AtoB', help='AtoB or BtoA')
parser.add_argument('--phase', dest='phase', default='train', help='train, test')
parser.add_argument('--save_epoch_freq', dest='save_epoch_freq', type=int, default=50, help='save a model every save_epoch_freq epochs (does not overwrite previously saved models)')
parser.add_argument('--save_latest_freq', dest='save_latest_freq', type=int, default=5000, help='save the latest model every latest_freq sgd iterations (overwrites the previous latest model)')
parser.add_argument('--print_freq', dest='print_freq', type=int, default=50, help='print the debug information every print_freq iterations')
parser.add_argument('--continue_train', dest='continue_train', type=bool, default=True, help='if continue training, load the latest model: 1: true, 0: false')
parser.add_argument('--serial_batches', dest='serial_batches', type=bool, default=False, help='f 1, takes images in order to make batches, otherwise takes them randomly')
parser.add_argument('--serial_batch_iter', dest='serial_batch_iter', type=bool, default=True, help='iter into serial image list')
parser.add_argument('--test_dir', dest='test_dir', default='./test', help='test sample are saved here')
parser.add_argument('--L1_lambda', dest='L1_lambda', type=float, default=1.0, help='weight on L1 term in objective')
parser.add_argument('--use_resnet', dest='use_resnet', type=bool, default=True, help='generation network using reidule block')
parser.add_argument('--use_lsgan', dest='use_lsgan', type=bool, default=True, help='gan loss defined in lsgan')
parser.add_argument('--max_size', dest='max_size', type=int, default=50, help='max size of image pool, 0 means do not use image pool')


args = parser.parse_args()

def main1(_):
    
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)
    sess_config = tf.ConfigProto(
        allow_soft_placement=True,
        gpu_options = tf.GPUOptions(allow_growth=True)
        )
    with tf.Session(config=sess_config) as sess:
        model = cyclegan(sess, args)
        model.train(args) if args.phase == 'train' \
            else model.test(args)

if __name__ == '__main__':
    tf.app.run(main1)
