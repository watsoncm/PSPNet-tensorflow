"""
Based on DrSleep's instructions here: https://gist.github.com/DrSleep/4bce37254c5900545e6b65f6a0858b9c
"""

import os
import numpy as np
import glob
import argparse
import random
import shutil
from tqdm import tqdm
from PIL import Image

DATA_DIRECTORY = '/home/ubuntu/data_road'
DATA_OUTPUT_DIRECTORY = '/home/ubuntu/kitti_road_seg'
DATA_TRAIN_LIST_PATH = '../list/kitti_train_list.txt'
DATA_VAL_LIST_PATH = '../list/kitti_val_list.txt'
TRAIN_VAL_SPLIT = 0.8

label_colors = [(0, 0, 0), (255, 0, 0), (255, 0, 255)]

parser = argparse.ArgumentParser(description='Preprocess the KITTI dataset.')


def get_arguments():
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the KITTI dataset.")
    parser.add_argument("--data-train-list-path", type=str, default=DATA_TRAIN_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--data-val-list-path", type=str, default=DATA_VAL_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--data-output-dir", type=str, default=DATA_OUTPUT_DIRECTORY,
                        help="Path to the output directory for the cleaned KITTI dataset.")
    return parser.parse_args()


def preprocess_gt(img):
    out_img = np.zeros(img.shape[:2])
    for i, label_color in enumerate(label_colors):
        eq_arr = (img == np.array(label_color).reshape(1, 1, 3))
        out_img[np.all(eq_arr, axis=2)] = i
    return out_img


def create_dataset(new_train_dir, new_test_dir, old_train_dir, old_train_gt_dir, old_test_dir):
    os.makedirs(new_train_dir)
    os.makedirs(new_test_dir)

    destinations = {old_train_dir:    new_train_dir, 
                    old_train_gt_dir: new_train_dir,
                    old_test_dir:     new_test_dir}

    for src_dir, dst_dir in destinations.iteritems():
        for img_path in tqdm(glob.glob(os.path.join(src_dir, '*.png'))):
            if src_dir == old_train_gt_dir:
                fname = os.path.basename(img_path)
                output_path = os.path.join(dst_dir, fname)
                img = np.array(Image.open(img_path))
                Image.fromarray(preprocess_gt(img)).convert('RGB').save(output_path, "PNG")
            else:
                shutil.copy(img_path, dst_dir)

    return new_train_dir, new_test_dir


def generate_lists(old_train_dir, data_train_list_path, data_val_list_path):
    train_val_imgs = os.listdir(old_train_dir)
    train_imgs = random.sample(train_val_imgs, int(len(train_val_imgs) * TRAIN_VAL_SPLIT))
    val_imgs = [img for img in train_val_imgs if img not in train_imgs]
    
    paths = {data_train_list_path: train_imgs,
             data_val_list_path: val_imgs}

    for path, imgs in paths.iteritems():
        lines = []
        for img in imgs:
            gt_img_parts = img.split('_')
            gt_img_parts.insert(1, 'road')
            gt_img = '_'.join(gt_img_parts)
            lines.append('{} {}'.format(img, gt_img))
    
        with open(path, 'w') as f:
            f.writelines(lines)


def main():
    """add docstring here"""
    args = get_arguments()
    new_train_dir = os.path.join(args.data_output_dir, 'train')
    new_test_dir = os.path.join(args.data_output_dir, 'test')
    old_train_dir = os.path.join(args.data_dir, 'training', 'image_2')
    old_train_gt_dir = os.path.join(args.data_dir, 'training', 'gt_image_2')
    old_test_dir = os.path.join(args.data_dir, 'testing', 'image_2')

    train_dir, test_dir = create_dataset(new_train_dir, new_test_dir, old_train_dir, old_train_gt_dir, old_test_dir)    
    generate_lists(old_train_dir, args.data_train_list_path, args.data_val_list_path)

if __name__ == '__main__':
    main()
