from __future__ import print_function
import argparse
import os
import sys
import time

from PIL import Image
import tensorflow as tf
import numpy as np

from tqdm import trange
from model import PSPNet101, PSPNet50
from tools import *

SNAPSHOT_DIR = './model'

param = {'crop_size': [720, 720],
         'num_classes': 3,
         'ignore_label': 0,
         'num_steps': 231,
         'model': PSPNet101,
         'data_dir': '/home/ubuntu/kitti_road_seg/train',
         'val_list': './list/kitti_train_list.txt'}

def get_arguments():
    parser = argparse.ArgumentParser(description="Reproduced PSPNet")

    parser.add_argument("--checkpoints", type=str, default=SNAPSHOT_DIR,
                        help="Path to restore weights.")
    parser.add_argument("--flipped-eval", action="store_true",
                        help="whether to evaluate with flipped img.")
    parser.add_argument("--dataset", type=str, default='kitti',
                        choices=['ade20k', 'cityscapes', 'kitti'])

    return parser.parse_args()

def load(saver, sess, ckpt_path):
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

def main():
    args = get_arguments()

    crop_size = param['crop_size']
    num_classes = param['num_classes']
    ignore_label = param['ignore_label']
    num_steps = param['num_steps']
    PSPNet = param['model']
    data_dir = param['data_dir']

    # Set placeholder 
    image_filename = tf.placeholder(dtype=tf.string)
    anno_filename = tf.placeholder(dtype=tf.string)

    # Read & Decode image
    img = tf.image.decode_image(tf.read_file(image_filename), channels=3)
    anno = tf.image.decode_image(tf.read_file(anno_filename), channels=1)
    img.set_shape([None, None, 3])
    anno.set_shape([None, None, 1])

    shape = tf.shape(img)
    h, w = (tf.maximum(crop_size[0], shape[0]), tf.maximum(crop_size[1], shape[1]))
    img = preprocess(img, h, w)

    # Create network.
    net = PSPNet({'data': img}, is_training=False, num_classes=num_classes)
    with tf.variable_scope('', reuse=True):
        flipped_img = tf.image.flip_left_right(tf.squeeze(img))
        flipped_img = tf.expand_dims(flipped_img, dim=0)
        net2 = PSPNet({'data': flipped_img}, is_training=False, num_classes=num_classes)

    raw_output = net.layers['conv6']

    # Do flipped eval or not
    if args.flipped_eval:
        flipped_output = tf.image.flip_left_right(tf.squeeze(net2.layers['conv6']))
        flipped_output = tf.expand_dims(flipped_output, dim=0)
        raw_output = tf.add_n([raw_output, flipped_output])

    # Scale feature map to image size, get prediction
    raw_output_up = tf.image.resize_bilinear(raw_output, size=[h, w], align_corners=True)
    raw_output_up = tf.image.crop_to_bounding_box(raw_output_up, 0, 0, shape[0], shape[1])
    raw_output_up = tf.argmax(raw_output_up, dimension=3)
    pred = tf.expand_dims(raw_output_up, dim=3)

    # Calculate mIoU
    pred_flatten = tf.reshape(pred, [-1,])
    raw_gt = tf.reshape(anno, [-1,])
    indices = tf.squeeze(tf.where(tf.not_equal(raw_gt, ignore_label)), 1)
    gt = tf.cast(tf.gather(raw_gt, indices), tf.int32)
    pred = tf.gather(pred_flatten, indices)

    # pred = tf.add(pred, tf.constant(1, dtype=tf.int64))
    mIoU, update_op = tf.contrib.metrics.streaming_mean_iou(pred, gt, num_classes=num_classes+1)

    # Set up tf session and initialize variables.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    global_init = tf.global_variables_initializer()
    local_init = tf.local_variables_initializer()
    sess.run(global_init)
    sess.run(local_init)

    restore_var = tf.global_variables()

    ckpt = tf.train.get_checkpoint_state(args.checkpoints)
    if ckpt and ckpt.model_checkpoint_path:
        loader = tf.train.Saver(var_list=restore_var)
        load_step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
        load(loader, sess, ckpt.model_checkpoint_path)
    else:
        print('No checkpoint file found.')


    file = open(param['val_list'], 'r') 
    for step in trange(num_steps, desc='evaluation', leave=True):
        f1, f2 = file.readline().split('\n')[0].split(' ')
        f1 = os.path.join(data_dir, f1)
        f2 = os.path.join(data_dir, f2)

        print('PRED:')
        print(sess.run(pred, feed_dict={image_filename: f1, anno_filename: f2}))
        _ = sess.run(update_op, feed_dict={image_filename: f1, anno_filename: f2})

    print('mIoU: {:04f}'.format(sess.run(mIoU)))

if __name__ == '__main__':
    main()
