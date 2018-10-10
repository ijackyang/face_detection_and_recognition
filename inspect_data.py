import os
import sys
import itertools
import math
import logging
import json
import re
import random
from collections import OrderedDict
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.patches import Polygon


ROOT_DIR = os.path.abspath("../../")

sys.path.append(ROOT_DIR)

import utils
import visualize
from visualize import display_images
import model as modellib
from model import log
import face

config = face.FaceConfig()
Face_DIR = os.path.join(ROOT_DIR,"dataset")

dataset = face.FaceDataset()
dataset.load_face(Face_DIR,"train")
dataset.prepare()

print("Image Count:{}".format(len(dataset.image_ids)))
print("Class COunt:{}".format(dataset.num_classes))
for i,info in enumerate(dataset.class_info):
    print("{:3}. {:50}".format(i,info['name']))

image_ids = np.random.choice(dataset.image_ids,2)
for image_id in image_ids:
    image = dataset.load_image(image_id)
    bbox,class_ids = dataset.load_bbox(image_id)
    print("image_id ",image_id,dataset.image_reference(image_id))
    log("image",image)
    log("class_ids",class_ids)
    log("bbox",bbox)
    visualize.display_instances(image,bbox,class_ids,dataset.class_names)


for image_id in np.random.choice(dataset.image_ids,64):
        image = dataset.load_image(image_id)
        bbox,class_ids = dataset.load_bbox(image_id)
        original_shape = image.shape
        image,window,scale,padding,_=utils.resize_image(
                image,
                min_dim=config.IMAGE_MIN_DIM,
                max_dim=config.IMAGE_MAX_DIM,
                mode=config.IMAGE_RESIZE_MODE)
        bbox = utils.resize_bbox(bbox,scale,padding)

        print("image_id:",image_id,dataset.image_reference(image_id))
        print("original shape",original_shape)
        log("image:",image)
        log("class_ids",class_ids)
        log("bbox",bbox)
        visualize.display_instances(image,bbox,class_ids,dataset.class_names)
    
