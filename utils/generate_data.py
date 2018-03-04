"""
Based on DrSleep's instructions here: https://gist.github.com/DrSleep/4bce37254c5900545e6b65f6a0858b9c
"""

import os
import glob
from PIL import image

DATA_DIRECTORY = '/home/ubuntu/data_road'
DATA_OUTPUT_DIRECTORY = '/home/ubuntu/kitti_road_seg'
DATA_TRAIN_LIST_PATH = '../list/kitti_train_list.txt'
DATA_VAL_LIST_PATH = '../list/kitti_val_list.txt'
TRAIN_VAL_SPLIT = 0.8

label_colours = [(0, 0, 0), (255, 0, 0), (255, 0, 255)]


def get_arguments():
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the KITTI dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--data-output-dir", type=str, default=DATA_OUTPUT_DIRECTORY,
                        help="Path to the output directory for the cleaned KITTI dataset.")


def preprocess_gt(img):
    out_img = np.zeros(img.shape[:2])
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] < len(label_colours):
                out_img[i, j] = label_colours[img[i, j]]
    return out_img


def create_dataset(new_train_dir, new_test_dir, old_train_dir, old_train_gt_dir, old_test_dir):
    os.makedirs(new_train_dir)
    os.makedirs(new_test_dir)

    destinations = {old_train_dir:    new_train_dir, 
                    old_train_gt_dir: new_train_dir,
                    old_test_dir:     new_test_dir}

    for src_dir, dst_dir in destinations.iteritems():
        for img_path in glob.glob(os.path.join(src_dir, '*.jpg')):
            if 
            if src_dir == old_train_gt_dir:
                fname = os.path.basename(img_path)
                output_path = os.path.join(dst_dir, fname)
                img = Image.open(img_path)
                preprocess_gt(img).save(output_path, "JPEG")
            else:
                shutil.copy(img_path, dst_dir)

    return new_train_dir, new_test_dir


def generate_lists(old_train_dir, data_train_list_path, data_test_list_path):
    train_val_imgs = os.listdir(old_train_dir)
    train_imgs = random.sample(train_val_imgs, len(train_val_imgs) * TRAIN_VAL_SPLIT)
    val_imgs = [img for img in train_val_imgs if img not in train_imgs]
    test_imgs = os.listdir(old_test_dir)
    
    paths = {train_imgs: data_train_list_path,
             test_imgs:  data_test_list_path}

    for imgs, path in paths.iteritems():
        lines = []
        for img in imgs:
            gt_img = '_'.join(img.split('_').insert(1, 'road'))
            lines.append('{} {}'.format(img, gt_img))
    
        with open(path) as f:
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
    generate_lists(old_train_dir, args.data_train_list_path, args.data_test_list_path)
