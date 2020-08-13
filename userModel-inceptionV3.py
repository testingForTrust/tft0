# Author: TFTAuthors
# This file is prepared to test(test) Google's inceptionV3 model trained on image net data based on Tensorflow(TF) Slim
# TF-Slim is a library that makes building, training and evaluation neural networks simple
# Source: https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim
# model under test: http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
# Details: https://github.com/tensorflow/models/tree/master/research/slim

import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.python.slim.nets.inception_v3 as inc_model
import tensorflow as tf
from keras.applications import inception_v3 as inc_net
import numpy as np
import os

# User imports TFT library uses APIs to generate test results
from tft.robustness_image import AdversarialInputs, AdversarialPatches
from tft.generalization_image import FourierRadialFiltering, ObjectRecognitionWeakSignals, RotationTranslation
from tft.interpretability_image import ModelInterpretation
from tft.performance_image.performance_metrics import PerformanceMetrics

# User creates a Model class that abstracts the model under test


class Model:
    
    def __init__(self, modelpath, model_file):
        """

        :param modelpath: a string; absolute path to the model (excluding file name)
        :param model_file: a string; file name like .meta or .h5
        """
        # The Model class must have following attributes
        self.image_size_height = 299  # Height of the image as per the model
        self.image_size_width = 299  # Width of the image as per the model
        self.num_channels = 3  # Number of channels in the image

        # sess: a tensorflow session with the model (graph plus weights) loaded by a user
        # logits: the model's output tensor i.e. the output of the softmax
        self.sess, self.logits = self.load_model(modelpath, model_file)
        # An input image placeholder
        self.x = self.sess.graph.get_tensor_by_name("x:0")

    def pre_process(imgs):
        """
        # The Model class must have a pre-process method
        :param imgs: A batch of images of shape (nSamples, h, w, c)
        :return: A batch of pre-processed images of shape (nSamples, h, w, c)
        """
        images_pre_processed = []
        for a_image in imgs:
            # a_image is of shape (dim1, dim2, channels)
            a_image = np.expand_dims(a_image, axis=0)  # line reshapes to (1, dim1, dim2, channels)
            # Pre process the input image as per inception_v3
            images_pre_processed.append(inc_net.preprocess_input(a_image))
        return np.vstack(images_pre_processed)  # Stack the images one below the other (rows)
        
    def rev_preprocess(self, im):
        """
        The model class must have reverse preprocess rev_preprocess() method

        :param im: A single image or a batch of images of respective shapes (h, w, c) or (nSamples, h, w, c)
        :return: A single image or a batch of images of respective shapes (h, w, c) or (nSamples, h, w, c)
        """
        im += 1
        im *= 128
        return im
        
    def predict(self, images):  # images will have values(pixel values) from 0 - 255
        """
        The Model class must have a predict method
        :param images: a list of image tensors of shape (nSamples, H, W, C) ; where H represents height, W represents
                width and C represents channels of an image respectively
        
        :return: array of arrays of predictions for each image sample.
               i.e. [[p(class0/image0),p(class1/image0),....,p(classN/image0)],
               [p(class0/image1),p(class1/image1),....,p(classN/image1)],.......,
               [p(class0/imageN),p(class1/imageN),....,p(classN/imageN)]]
        """
        probabilities = tf.nn.softmax(self.logits)
        print(images.shape)
        pre_processed_images = self.pre_process(images)
        print(pre_processed_images.shape)
        
        return self.sess.run(probabilities, feed_dict={self.x: pre_processed_images})
    
    def load_model(self, path_to_model, model_file_name):
        """
        The Model class must have a method to load the model
        :param path_to_model: a string ; absolute path to the model
        :return: 1. a tensorflow session object with model (graph and weights) loaded
                 2. logits - the model's output tensor i.e. the output of the softmax
        """
        sess = tf.Session()
        saver = tf.train.import_meta_graph(os.path.join(path_to_model, model_file_name))
        LOGITS_TENSOR_NAME = 'InceptionV3/Logits/SpatialSqueeze:0'
        saver.restore(sess, tf.train.latest_checkpoint(path_to_model))
        logits = tf.get_default_graph().get_tensor_by_name(LOGITS_TENSOR_NAME)
        print("Checkpoint loaded..")
        print("logits..."+str(logits))
        return sess, logits


# Define paths to the model, test samples, image vs label csv, path to save results and a json that has index-class
# mapping. NOTE: path to save results must be the following path under tomcat webapps folder.
# <tomcat webapps folder>\teachntest\assets\results

PATH_TO_THE_MODEL = r''  # Absolute path to model
IMAGE_SAMPLES_FOLDER = r''  # Absolute path to folder where test data/Image samples are present
IMAGE_VS_LABELS_CSV = r''  # Absolute path to image-label csv
PATH_TO_SAVE_RESULTS = r''  # Absolute path to save results - must be <tomcat webapps folder>\teachntest\assets\results
PATH_TO_JSON_INDEX_CLASS_MAPPING = r''  # a local absolute path to a .json file that contains the index-class mapping
PROJECT_NAME = r''  # A string which represents a name under which an test/test is performed
model_file_name = r''  """A string which is the file name along with the extension of the model under test.
                          It must be one of .h5, .meta and .pb file names."""

# instantiate the Model class and start testing the model against different methods..
model = Model(os.path.join(PATH_TO_THE_MODEL, model_file_path))
test0 = AdversarialPatches(model, IMAGE_SAMPLES_FOLDER, IMAGE_VS_LABELS_CSV, PATH_TO_SAVE_RESULTS, PROJECT_NAME,
                            PATH_TO_JSON_INDEX_CLASS_MAPPING, 1001, PATH_TO_THE_MODEL, model_file_name, 'x:0',
                            'InceptionV3/Logits/SpatialSqueeze:0', 13, (-1, 1), 'rectangle', 4, learning_rate=5.0)
result = test0.run()
print(result)                         
test1 = AdversarialInputs(model, IMAGE_SAMPLES_FOLDER, IMAGE_VS_LABELS_CSV, PATH_TO_SAVE_RESULTS,
                           PROJECT_NAME, PATH_TO_JSON_INDEX_CLASS_MAPPING, 1001, threshold=0.1)
result = test1.fgsm(0.015)  # epsilon value
print(result)
result = test1.cw()  # Carlini & Wagner Method: can take optional learning rate & number of iterations
print(result)

test2 = FourierRadialFiltering(model, IMAGE_SAMPLES_FOLDER, IMAGE_VS_LABELS_CSV, PATH_TO_SAVE_RESULTS,
                                PROJECT_NAME, PATH_TO_JSON_INDEX_CLASS_MAPPING, radius=0.4, threshold=0.1)
result = test2.run()
print(result)

test3 = ModelInterpretation(model, IMAGE_SAMPLES_FOLDER, IMAGE_VS_LABELS_CSV, PATH_TO_SAVE_RESULTS,
                             PROJECT_NAME, PATH_TO_JSON_INDEX_CLASS_MAPPING, num_features=5, num_samples=200,
                             hide_rest=True)
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

# Performance Metrics
pm = PerformanceMetrics(model, "x:0", PATH_TO_THE_MODEL, model_file_name, True, PATH_TO_SAVE_RESULTS, PROJECT_NAME,
                        IMAGE_SAMPLES_FOLDER, IMAGE_VS_LABELS_CSV, model.image_size_height, model.image_size_width)
pm.compute()
