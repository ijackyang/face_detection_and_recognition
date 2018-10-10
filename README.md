# face_detection_and_recognition
end-to-end face detection and recognition training based on faster RCNN.

This project's code is based on the mask-rcnn: https://github.com/matterport/Mask_RCNN.
But the mask prediction part is removed, but I add the spatial network transform and the face recognition part. 
For the training part, I created the detection and recognition training dataset. The face with known id in the picture is detected and then 16 pictures are stitched together to form one big picture. Then the end-to-end face detection and recognition is trained on this dataset. This is not good for the detection part, better dataset should be provided for this end-to-end training.

Further demonstration will be added to this repository.
