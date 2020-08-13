# author: TFTAuthors
# TFT on Saved ResNet model (imagenet dataset)
# Resnet50 architecture trainied on ImageNet data
# Link to the model: http://download.tensorflow.org/models/official/20181001_resnet/savedmodels/resnet_v2_fp32_savedmodel_NCHW.tar.gz

import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
import os
import numpy as np
import pandas as pd
from tft.robustness_image import AdversarialInputs, AdversarialPatches
from tft.generalization_image import FourierRadialFiltering, ObjectRecognitionWeakSignals, RotationTranslation
from tft.interpretability_image import ModelInterpretation
from tft.performance_image.performance_metrics import PerformanceMetrics
from cleverhans import utils_tf
from PIL import Image
import time
time.time()
start_time = time.time()
print(start_time)

class Model():

    def __init__(self, modelpath):
        self.image_size_height = 224
        self.image_size_width = 224
        self.num_channels = 3
        self.sess, self.inputImage, self.output1 = self.load_model(modelpath)
        self.x = self.inputImage
        self.logits = self.output1
        self.y = tf.placeholder(tf.float32, shape=(None, 1001))
        self.loss = utils_tf.model_loss(self.y, self.logits, mean=False)

    def pre_process(self, imgs):
        images_pre_processed = []
        for f in imgs:
            x = np.expand_dims(f, axis=0)
            x = preprocess_input(x)  # by default, uses mode='caffe'. The rev_preprocess() method below was worked out accordingly.
            images_pre_processed.append(x)
        return np.vstack(images_pre_processed)
		
	def rev_preprocess(self,im):
        # im will be in the form BGR
        mean = [103.939, 116.779, 123.68]
        im[...,0]+= mean[0]
        im[...,1]+= mean[1]
        im[...,2]+= mean[2]
        # convert back to RGB
        if len(im.shape) == 4: # For a batch of images
            im = im[:,:,:,(2,1,0)]
        else:
            im = im[:,:,(2,1,0)] # For a single image
        return im

    def predict(self, images):
        pre_processed_images = self.pre_process(images)
        input_img = self.inputImage
        probab = self.output1
        return self.sess.run(probab, feed_dict={input_img: pre_processed_images})

    def load_model(self, path_to_model):

        sess=tf.Session()
        inputImage = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
        model = tf.saved_model.loader.load(sess=sess, export_dir= path_to_model, tags=[tag_constants.SERVING], input_map={'input_tensor:0':  inputImage})
        output1 = sess.graph.get_tensor_by_name('softmax_tensor:0')


        print(inputImage)
        print(output1)

        return sess, inputImage, output1

# Define paths to the model, test samples, image vs label csv, path to save results and a json that has index-class
# mapping. NOTE: path to save results must be the following path under tomcat webapps folder.
# <tomcat webapps folder>\teachntest\assets\results

PATH_TO_THE_MODEL = r''  # Absolute path to model
IMAGE_SAMPLES_FOLDER = r''  # Absolute path to folder where test data/Image samples are present
IMAGE_VS_LABELS_CSV = r''  # Absolute path to image-label csv
PATH_TO_SAVE_RESULTS = r''  # Absolute path to save results - must be <tomcat webapps folder>\teachntest\assets\results
PATH_TO_JSON_INDEX_CLASS_MAPPING = r''  # a local absolute path to a .json file that contains the index-class mapping
PROJECT_NAME = r''  # A string which represents a name under which an test/test is performed
model_file_name = r'' # A string which is the file name along with the extension of the model under test. A .pb file name in this case.

# instantiate the Model class and start testing the model against different methods..
model = Model(PATH_TO_THE_MODEL)

test0 = AdversarialPatches( model,IMAGE_SAMPLES_FOLDER, IMAGE_VS_LABELS_CSV, PATH_TO_SAVE_RESULTS, PROJECT_NAME,
                            PATH_TO_JSON_INDEX_CLASS_MAPPING, 1001, PATH_TO_THE_MODEL,model_file_name,'input_tensor:0',
                            'softmax_tensor:0',16, (-103, 131), 'rectangle', 64, learning_rate=5.0, target_label='tractor')
result = test0.run()
print(result1)							

test1 = AdversarialInputs(model, IMAGE_SAMPLES_FOLDER, IMAGE_VS_LABELS_CSV, PATH_TO_SAVE_RESULTS,
                           PROJECT_NAME, PATH_TO_JSON_INDEX_CLASS_MAPPING, 1000, threshold=0.1)
result = test1.fgsm(0.015)  # epsilon value
print(result)

result = test1.cw() # Carlini & Wagner Method: can take optional learning rate & number of iterations
print(result)

test2 = FourierRadialFiltering(model, IMAGE_SAMPLES_FOLDER, IMAGE_VS_LABELS_CSV, PATH_TO_SAVE_RESULTS,
                                PROJECT_NAME, PATH_TO_JSON_INDEX_CLASS_MAPPING, radius=0.4, threshold=0.1)
result = test2.run()
print(result)

test3 = ModelInterpretation(model, IMAGE_SAMPLES_FOLDER, IMAGE_VS_LABELS_CSV, PATH_TO_SAVE_RESULTS,
                             PROJECT_NAME, PATH_TO_JSON_INDEX_CLASS_MAPPING, num_features=5, num_samples=200, hide_rest=True)
result = test3.run()
print(result)

test4 = ObjectRecognitionWeakSignals(model, IMAGE_SAMPLES_FOLDER, IMAGE_VS_LABELS_CSV, PATH_TO_SAVE_RESULTS,
                                      PROJECT_NAME, PATH_TO_JSON_INDEX_CLASS_MAPPING, threshold=0.1)

result = test4.generate_gray_scale()
print(result)
result = test4.generate_low_contrast(contrast_level_1=0.6)
print(result)
result = test4.generate_noisy(noise_width=0.1, contrast_level_2=0.3)
print(result)

"""
- reach: float, controlling the strength of the manipulation
- coherence: a float within [0, 1] with 1 = full coherence
- grain: float, controlling how fine-grained the distortion is
"""
result = test4.generate_eidolon(grain=10.0, coherence=1.0, reach=2.0)
print(result)

test5 = RotationTranslation(model, IMAGE_SAMPLES_FOLDER, IMAGE_VS_LABELS_CSV, PATH_TO_SAVE_RESULTS,
                             PROJECT_NAME, PATH_TO_JSON_INDEX_CLASS_MAPPING, threshold=0.1)
result = test5.run()
print(result)

# PerformanceMetrics
pm = PerformanceMetrics(model, "input_tensor:0", PATH_TO_THE_MODEL, model_file_name, False, PATH_TO_SAVE_RESULTS, PROJECT_NAME, IMAGE_SAMPLES_FOLDER, IMAGE_VS_LABELS_CSV, model.image_size_height, model.image_size_width)
pm.compute()