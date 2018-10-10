
Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=imagenet

    # Apply color splash to an image
    python3 balloon.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 balloon.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from config import Config
import model as modellib
import utils as utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class FaceConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "face"

    # We use a GPU with 32GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + face

    # Backbone network architecture
    # Supported values are: resnet50, resnet101
    BACKBONE = "resnet50"

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 4096

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

    # Learning rate and momentum
    # The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes
    # weights to explode. Likely due to differences in optimzer
    # implementation.
    LEARNING_RATE = 0.01
    LEARNING_MOMENTUM = 0.9

    ## Maximum num of faces in one image
    NUM_FACES = 10575

    # 0~1 learning rate for learning center loss
    ALPHA = 0.5

    # loss weights
    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 1.,
        "recog_class_loss": 0.,
        "center_loss": 0.
    }






############################################################
#  Dataset
############################################################

class FaceDataset(utils.Dataset):


    def __init__(self,dataset='detection'):
        if dataset=='detection':
                lines = [line.rstrip('\n') for line in open('../../detection_dataset/wider_face_train_bbx_gt.txt')]+\
                        [line.rstrip('\n') for line in open('../../detection_dataset/wider_face_val_bbx_gt.txt')]
        elif dataset=='recognition':
                lines = [line.rstrip('\n') for line in open('../../recognition_dataset/bbox_ids_train.txt')]+\
                        [line.rstrip('\n') for line in open('../../recognition_dataset/bbox_ids_val.txt')]
        i = 0
        bbox_dict = {}
        while(i<len(lines)):
                image_id = lines[i]
                num_of_bbox = int(lines[i+1])
                bbox_list = []
                for ii in range(i+2,i+2+num_of_bbox):
                        bbox = [int(bbox_c) for bbox_c in lines[ii].split()]
                        if dataset=='detection':
                                bbox[4] = 1
                        if(bbox[2]>1 and bbox[3]>1):
                                bbox_list.append(bbox)
                i = i+2+num_of_bbox
                bbox_dict[image_id]=bbox_list
        self.bbox_dict = bbox_dict
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        self._image_ids = []
        self.image_info = []
        self.source_class_ids = {}

    def load_face(self, dataset_dir, subset):
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("face", 1, "face")

        # Train or validation dataset?
        assert subset in ["train", "val","test"]
        dataset_dir = os.path.join(dataset_dir, subset)

        #load image
        #image stored in this form;
        #image_category/image_name
        face_dirs = next(os.walk(dataset_dir))[1]
        face_ids  = []
        for face_dir in face_dirs:
                fids = next(os.walk(os.path.join(dataset_dir,face_dir)))[2]
                for fid in fids:
                        face_ids.append(os.path.join(face_dir,fid))


        for face_id in face_ids:
                image_path = os.path.join(dataset_dir,face_id)
                image = skimage.io.imread(image_path)
                height,width = image.shape[:2]
                if(len(self.bbox_dict[face_id])!=0): 
                        self.add_image(
                                "face",
                                image_id = face_id,
                                path     = image_path,
                                width    = width,
                                height   = height)         

    def load_bbox(self,image_id):
        "load image bounding box"
        bbox_list = np.array([np.array([bbox[1],bbox[0],bbox[1]+bbox[3],bbox[0]+bbox[2]]) for bbox in self.bbox_dict[self.image_info[image_id]['id']]])
        class_ids = np.ones([len(bbox_list)], dtype=np.int32)
        return bbox_list,class_ids

    def load_face_ids(self,config,image_id):
        face_ids = np.array([bbox[4]-1 for bbox in self.bbox_dict[self.image_info[image_id]['id']]])
        #face_ids = np.eye(config.NUM_FACES)[face_ids]
        return face_ids

    def show_gt_bbox(self,image_id):
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        fig,ax = plt.subplots(1)

        image = skimage.io.imread(os.path.join('../../dataset/train',image_id))
        ax.imshow(image)
        for bbox in self.bbox_dict[image_id]:
                rect = patches.Rectangle((bbox[0],bbox[1]),bbox[2],bbox[3],edgecolor='r',facecolor='none')
                ax.add_patch(rect)
        plt.show()

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "balloon":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

def train(model,dataset='detection'):
    """Train the model."""
    # Training dataset.
    dataset_train = FaceDataset(dataset=dataset)
    dataset_train.load_face(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = FaceDataset(dataset=dataset)
    dataset_val.load_face(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=375,
                layers='detection')



def add_bbox_on_image(model,image_path):
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        image = skimage.io.imread(image_path)
        r = model.detect([image], verbose=1)[0]
        rois = r['rois']
        face_ids = r['face_ids']
        fig,ax = plt.subplots(1)
        ax.imshow(image)
        for i in range(rois.shape[0]):
                bbox = rois[i]
                face_id = face_ids[i]
                print("bbox {}".format(bbox))
                print(" face_id:{}\n".format(face_id))
                rect = patches.Rectangle((bbox[1],bbox[0]),bbox[3]-bbox[1],bbox[2]-bbox[0],edgecolor='r',facecolor='none')
                ax.add_patch(rect)
        plt.show()

#def color_splash(image, mask):
#    """Apply color splash effect.
#    image: RGB image [height, width, 3]
#    mask: instance segmentation mask [height, width, instance count]
#
#    Returns result image.
#    """
#    # Make a grayscale copy of the image. The grayscale copy still
#    # has 3 RGB channels, though.
#    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
#    # We're treating all instances as one, so collapse the mask into one layer
#    mask = (np.sum(mask, -1, keepdims=True) >= 1)
#    # Copy color pixels from the original color image where mask is set
#    if mask.shape[0] > 0:
#        splash = np.where(mask, image, gray).astype(np.uint8)
#    else:
#        splash = gray
#    return splash


#def detect_and_color_splash(model, image_path=None, video_path=None):
#    assert image_path or video_path
#
#    # Image or video?
#    if image_path:
#        # Run model detection and generate the color splash effect
#        print("Running on {}".format(args.image))
#        # Read image
#        image = skimage.io.imread(args.image)
#        # Detect objects
#        r = model.detect([image], verbose=1)[0]
#        # Color splash
#        splash = color_splash(image, r['masks'])
#        # Save output
#        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
#        skimage.io.imsave(file_name, splash)
#    elif video_path:
#        import cv2
#        # Video capture
#        vcapture = cv2.VideoCapture(video_path)
#        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
#        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
#        fps = vcapture.get(cv2.CAP_PROP_FPS)
#
#        # Define codec and create video writer
#        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
#        vwriter = cv2.VideoWriter(file_name,
#                                  cv2.VideoWriter_fourcc(*'MJPG'),
#                                  fps, (width, height))
#
#        count = 0
#        success = True
#        while success:
#            print("frame: ", count)
#            # Read next image
#            success, image = vcapture.read()
#            if success:
#                # OpenCV returns images as BGR, convert to RGB
#                image = image[..., ::-1]
#                # Detect objects
#                r = model.detect([image], verbose=0)[0]
#                # Color splash
#                splash = color_splash(image, r['masks'])
#                # RGB -> BGR to save image to video
#                splash = splash[..., ::-1]
#                # Add image to video writer
#                vwriter.write(splash)
#                count += 1
#        vwriter.release()
#    print("Saved to ", file_name)


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Faster R-CNN to detect faces.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/face/dataset/",
                        help='Directory of the Face dataset')
    parser.add_argument('--subset',required=True,
                         metavar="detection or recognition",
                         help='choose sub dataset to train')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = FaceConfig()
    else:
        class InferenceConfig(FaceConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.FasterRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.FasterRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()[1]
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train" and args.subset=="detection":
        train(model,dataset='detection')
    elif args.command == "train" and args.subset=="recognition":
        train(model,dataset="recognition")
    elif args.command == "detect":
        #dataset_train = FaceDataset()
        #dataset_train.load_face(args.dataset, "train")
        #dataset_train.prepare()
        #dataset_train.show_gt_bbox('0--Parade/0_Parade_marchingband_1_778.jpg')
        add_bbox_on_image(model,args.image)
        #detect_and_color_splash(model, image_path=args.image,video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
