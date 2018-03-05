from __future__ import print_function

import argparse
import os
import sys
import time
import tensorflow as tf
import numpy as np
from scipy import misc

from model import PSPNet101, PSPNet50
from tools import *

param = {'crop_size': [720, 720],
         'num_classes': 3,
         'model': PSPNet101}

SAVE_DIR = './output/'
SNAPSHOT_DIR = './model/'

def get_arguments():
    parser = argparse.ArgumentParser(description="Reproduced PSPNet")
    parser.add_argument("--img-path", type=str, default='',
                        help="Path to the RGB image file.")
    parser.add_argument("--checkpoints", type=str, default=SNAPSHOT_DIR,
                        help="Path to restore weights.")
    parser.add_argument("--save-dir", type=str, default=SAVE_DIR,
                        help="Path to save output.")
    parser.add_argument("--flipped-eval", action="store_true",
                        help="whether to evaluate with flipped img.")
    parser.add_argument("--dataset", type=str, default='kitti',
                        choices=['ade20k', 'cityscapes', 'kitti'])

    return parser.parse_args()

def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)

    if not os.path.exists(logdir):
        os.makedirs(logdir)
    saver.save(sess, checkpoint_path, global_step=step)
    print('The checkpoint has been created.')

def load(saver, sess, ckpt_path):
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

def main():
    args = get_arguments()

    # load parameters
    crop_size = param['crop_size']
    num_classes = param['num_classes']
    PSPNet = param['model']

    # preprocess images
    img, filename = load_img(args.img_path)
    img_shape = tf.shape(img)
    h, w = (tf.maximum(crop_size[0], img_shape[0]), tf.maximum(crop_size[1], img_shape[1]))
    img = preprocess(tf.Print(img, [h, w]), h, w)

    # Create network.
    net = PSPNet({'data': tf.Print(img, [tf.shape(img)])}, is_training=False, num_classes=num_classes)
    with tf.variable_scope('', reuse=True):
        flipped_img = tf.image.flip_left_right(tf.squeeze(img))
        flipped_img = tf.expand_dims(flipped_img, dim=0)
        net2 = PSPNet({'data': flipped_img}, is_training=False, num_classes=num_classes)

    raw_output = tf.Print(net.layers['conv6'], ['whoaaa']) 
    # Do flipped eval or not
    if args.flipped_eval:
        flipped_output = tf.image.flip_left_right(tf.squeeze(net2.layers['conv6']))
        flipped_output = tf.expand_dims(flipped_output, dim=0)
        raw_output = tf.add_n([raw_output, flipped_output])

    # Predictions.
    raw_output_up_a = tf.image.resize_bilinear(tf.Print(raw_output, ['waluigi:', tf.shape(raw_output)]), size=[h, w], align_corners=True)
    raw_output_up_b  = tf.image.crop_to_bounding_box(tf.Print(raw_output_up_a, ['wahhhh:', tf.shape(raw_output_up_a), img_shape]), 0, 0, img_shape[0], img_shape[1])
    raw_output_up = tf.argmax(raw_output_up_b, dimension=3)
    #raw_output_up = tf.argmax(raw_output, dimension=3)
    pred = decode_labels(tf.Print(raw_output_up, ['3:', img_shape, tf.unique(tf.reshape(raw_output_up, [-1]))[0]]), img_shape, num_classes)
    
    # Init tf Session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()

    sess.run(init)
    
    restore_var = tf.global_variables()
    
    ckpt = tf.train.get_checkpoint_state(args.checkpoints)
    if ckpt and ckpt.model_checkpoint_path:
        loader = tf.train.Saver(var_list=restore_var)
        load_step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
        load(loader, sess, ckpt.model_checkpoint_path)
    else:
        print('No checkpoint file found.')
    
    """
    ra = sess.run(tf.squeeze(tf.argmax(raw_output_up_a, dimension=3)))
    print('RAW A:')
    print(ra)
    print(ra.shape)
    print((ra * 256).shape)
    print(np.bincount(ra.flatten()))
    misc.imsave('a.png', ra * 256)
    """

    rb = sess.run(tf.squeeze(tf.argmax(raw_output_up_a, dimension=3)))
    print('RAW B:')
    print(rb)
    print(rb.shape)
    print((rb * 256).shape)
    print(np.bincount(rb.flatten()))
    misc.imsave('b.png', rb * 256)



    print('asdf')
    preds = sess.run(tf.Print(pred, ['pred:', tf.unique_with_counts(tf.reshape(tf.argmax(pred, -1), [-1]))[2]]))
    # print(preds)
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    misc.imsave(args.save_dir + filename, preds[0])
    
if __name__ == '__main__':
    main()
