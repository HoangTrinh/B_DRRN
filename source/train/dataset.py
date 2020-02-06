import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import random
import glob
import io
import numpy as np
import PIL.Image as pil_image
import shutil
import cv2
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.compat.v1.enable_eager_execution(config=config)



class mDataset(object):
    def __init__(self, images_dir, parts_dir, labels_dir, num_channels):
        self.image_files = sorted(glob.glob(images_dir + '/*'))
        self.parts_dir=parts_dir
        self.labels_dir = labels_dir
        self.num_channels = num_channels

    def __getitem__(self, idx):
        input = pil_image.open(self.image_files[idx])
        part =  pil_image.open(os.path.join(self.parts_dir, os.path.basename(self.image_files[idx])))
        label = pil_image.open(os.path.join(self.labels_dir, os.path.basename(self.image_files[idx])))

        input = np.array(input).astype(np.float32)
        part = np.array(part).astype(np.float32)
        label = np.array(label).astype(np.float32)
        #print(input.shape)

        if self.num_channels == 1:
            input = np.expand_dims(input, axis=0)
            part = np.expand_dims(part, axis=0)
            label = np.expand_dims(label, axis=0)
        else:
            input = np.transpose(input, axes=[2, 0, 1])
            part = np.transpose(part, axes=[2, 0, 1])
            label = np.transpose(label, axes=[2, 0, 1])


        # normalization
        input /= 255.0
        part /= 255.0
        label /= 255.0

        return input, part, label

    def __len__(self):
        return len(self.image_files)
