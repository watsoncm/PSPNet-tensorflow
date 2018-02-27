from __future__ import print_function
import argparse
import os.path
import numpy as np
import tensorflow as tf

DATA_DIRECTORY = '/home/ubuntu/kitti_road_seg/train/'
DATA_LIST_PATH = '/home/ubuntu/PSPNet-tensorflow/list/kitti_train_list.txt'

def get_arguments():
    parser = argparse.ArgumentParser(description="Find image averages")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    return parser.parse_args()

def get_image_mean(fname):
    img_data = tf.read_file(fname)
    img = tf.image.decode_jpeg(img_data, channels=3) 
    return tf.reduce_mean(tf.cast(img, tf.float32), axis=[0, 1])

def get_train_mean():
    """Create the model and start the training."""
    args = get_arguments()
    file_str = tf.read_file(DATA_LIST_PATH)
    text = tf.sparse_tensor_to_dense(tf.string_split([file_str], '\n'), default_value='')
    lines = tf.squeeze(tf.reshape(text, [1, -1]))
    imgs = tf.sparse_tensor_to_dense(tf.string_split(lines, ' '), default_value='')[:, 0]
    full_paths = tf.string_join([DATA_DIRECTORY, imgs])
    avgs = tf.map_fn(get_image_mean, full_paths, dtype=np.float32)
    return tf.reduce_mean(avgs, axis=[0])

def main():
    mean = get_train_mean()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        mean_value = sess.run(mean)
   
    format_string = 'IMG_MEAN = np.array(({}, {}, {}), dtype=np.float32)'
    print('\nAdd this line to train.py:')
    print(format_string.format(*mean_value))
    
if __name__ == '__main__':
    main()
