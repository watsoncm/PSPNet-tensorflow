# PSPNet_tensorflow

UPDATE: progress so far has been slow but steady -- getting these results took a bit longer than expected. A series of issues with getting the PSPNet-tensorflow model running with KITTI have been recently resolved, resulting in a mIoU of 42.57% for the training set and 42.46% for validation. This model was transfer learned for 3.5 epochs onto the KITTI road dataset with an 80/20 hold-out cross-validation split from the Cityscapes baseline provided with the parent repository of this repo, keeping the same learning rate schedule but using Adam for optimization instead of momentum. The underlying model is PSPNet101, a pyramid spatial pooling module-based architecture built from ResNet101. Additional modification of the learning rate schedule (or even not using one at all) and random search for hyperparameters could push these benchmarks up even further, although at this point our team has decided to focus on developing a PSPNet/DeepLab-based architecture specifically for KITTI (more on this later).

Although we do not have an entirely self-designed model yet, we believe that this indicates the viability of transfer learning from other larger datasets like CityScapes or ADE20K onto KITTI. The primary issue with the framework given by this repo's parent repository is that PSPNet is designed for significantly larger images. As a work-around, we used bilinear upsampling to double the image size (and nearest neighbor upsampling for the ground truth labels), then transfer learned the last layers of PSPNet101, but of course such an approach has many inherent weaknesses (i.e. upsampling artifacts potentially affecting first-layer convolution features) and as such we hope to design a much model specifically for KITTI.

The current plan is to perform net surgery on PSPNet50 as described in the original paper in order to allow direct input of smaller images. From there, transfer learning will be performed again, this time on our own framework, the data will be split the same as was done here, and a random search will be performed over the hyperparameter space in order to find out the best way to transfer learn our model. From there, we hope to search the literature in order to find ways to incorporate a classical CV prior. Our hope is that by doing so, we will be able to significantly improve performance on KITTI given the small size of the dataset. Shape priors already appear to be part of the literature, e.g. [here](https://pdfs.semanticscholar.org/ace6/feb2f700a0dc1c4f3f30817d196412b430e9.pdf), but seem to be primarily focused on object segmentation rather than scene parsing. 

Additionally, for the sake of possible weakly supervised learning later on, a corpus of road images has been collected for potential future hand-labeling.

The remainder of the README is from the parent repo.

## Introduction
  This is an implementation of PSPNet in TensorFlow for semantic segmentation on the [cityscapes](https://www.cityscapes-dataset.com/) dataset. We first convert weight from [Original Code](https://github.com/hszhao/PSPNet) by using [caffe-tensorflow](https://github.com/ethereon/caffe-tensorflow) framework.

## Update:
#### 2018/01/24:
1. `Support evaluation code for ade20k dataset`

#### 2018/01/19:
1. `Support inference phase for ade20k dataset` using model of pspnet50 (convert weights from original author)
2. Using `tf.matmul` to decode label, so as to improve the speed of inference.
#### 2017/11/06:
`Support different input size` by padding input image to (720, 720) if original size is smaller than it, and get result by cropping image in the end.
#### 2017/10/27: 
Change bn layer from `tf.nn.batch_normalization` into `tf.layers.batch_normalization` in order to support training phase. Also update initial model in Google Drive.

## Install 
Get restore checkpoint from [Google Drive](https://drive.google.com/drive/folders/1S90PWzXEX_GNzulG1f2eTHvsruITgqsm?usp=sharing) and put into `model` directory. Note: Select the checkpoint corresponding to the dataset.

## Inference
To get result on your own images, use the following command:
```
python inference.py --img-path=./input/test.png --dataset cityscapes  
```
Inference time:  ~0.6s 

Options:
```
--dataset cityscapes or ade20k
--flipped-eval 
--checkpoints /PATH/TO/CHECKPOINT_DIR
```
## Evaluation
### Cityscapes
Perform in single-scaled model on the cityscapes validation datase.

| Method | Accuracy |  
|:-------:|:----------:|
| Without flip| **76.99%** |
| Flip        | **77.23%** |

### ade20k
| Method | Accuracy |  
|:-------:|:----------:|
| Without flip| **40.00%** |
| Flip        | **40.67%** |

To re-produce evluation results, do following steps:
1. Download [Cityscape dataset](https://www.cityscapes-dataset.com/) or [ADE20k dataset](http://sceneparsing.csail.mit.edu/) first. 
2. change `data_dir` to your dataset path in `evaluate.py`:
```
'data_dir': ' = /Path/to/dataset'
```
3. Run the following command: 
```
python evaluate.py --dataset cityscapes
```
List of Args:
```
--dataset - ade20k or cityscapes
--flipped-eval  - Using flipped evaluation method
--measure-time  - Calculate inference time
```

## Image Result
### cityscapes
Input image                |  Output image
:-------------------------:|:-------------------------:
![](https://github.com/hellochick/PSPNet_tensorflow/blob/master/input/test_1024x2048.png)  |  ![](https://github.com/hellochick/PSPNet_tensorflow/blob/master/output/test_1024x2048.png)
![](https://github.com/hellochick/PSPNet_tensorflow/blob/master/input/test_720x720.png)  |  ![](https://github.com/hellochick/PSPNet_tensorflow/blob/master/output/test_720x720.png)

### ade20k
Input image                |  Output image
:-------------------------:|:-------------------------:
![](https://github.com/hellochick/PSPNet_tensorflow/blob/master/input/indoor_2.jpg)  |  ![](https://github.com/hellochick/PSPNet_tensorflow/blob/master/output/indoor_2.jpg)

### real world
Input image                |  Output image
:-------------------------:|:-------------------------:
![](https://github.com/hellochick/PSPNet_tensorflow/blob/master/input/indoor_1.jpg)  |  ![](https://github.com/hellochick/PSPNet_tensorflow/blob/master/output/indoor_1.jpg)

### Citation
    @article{zhao2017pspnet,
      author = {Hengshuang Zhao and
                Jianping Shi and
                Xiaojuan Qi and
                Xiaogang Wang and
                Jiaya Jia},
      title = {Pyramid Scene Parsing Network},
      booktitle = {Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
      year = {2017}
    }
Scene Parsing through ADE20K Dataset. B. Zhou, H. Zhao, X. Puig, S. Fidler, A. Barriuso and A. Torralba. Computer Vision and Pattern Recognition (CVPR), 2017. (http://people.csail.mit.edu/bzhou/publication/scene-parse-camera-ready.pdf)

    @inproceedings{zhou2017scene,
        title={Scene Parsing through ADE20K Dataset},
        author={Zhou, Bolei and Zhao, Hang and Puig, Xavier and Fidler, Sanja and Barriuso, Adela and Torralba, Antonio},
        booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
        year={2017}
    }
    
Semantic Understanding of Scenes through ADE20K Dataset. B. Zhou, H. Zhao, X. Puig, S. Fidler, A. Barriuso and A. Torralba. arXiv:1608.05442. (https://arxiv.org/pdf/1608.05442.pdf)

    @article{zhou2016semantic,
      title={Semantic understanding of scenes through the ade20k dataset},
      author={Zhou, Bolei and Zhao, Hang and Puig, Xavier and Fidler, Sanja and Barriuso, Adela and Torralba, Antonio},
      journal={arXiv preprint arXiv:1608.05442},
      year={2016}
    }
