# author: TFTAuthors
# This file is prepared to test(test) mobilenet model trained on four traffic signs

import tensorflow as tf

import numpy as np
# User imports our library
from tft.robustness_image import AdversarialInputs, AdversarialPatches
from tft.generalization_image import FourierRadialFiltering, ObjectRecognitionWeakSignals, RotationTranslation
from tft.interpretability_image import ModelInterpretation
from keras.models import load_model
from keras import backend as K
from cleverhans import utils_tf
from tft.performance_image.performance_metrics import PerformanceMetrics


class Model:
    """
    Define a Model to be loaded by the user.
    param logits: the model's output tensor i.e. the output of the softmax
    param sess: a tensorflow session with the checkpoint/model loaded by a user
    """
    def __init__(self, modelpath, model_file):
        self.image_size_height = 224
        self.image_size_width = 224
        self.num_channels = 3
        self.sess, self.model, self.logits = self.load_model(modelpath, model_file)
        self.y = tf.placeholder(tf.float32, shape=(None, 4))
        self.loss = utils_tf.model_loss(self.y, self.model.outputs[0], mean=False)
        self.x = self.model.inputs[0]  # input tensor

    def preprocess_input(im):
        im2 = im.astype(np.float32)
        im2 /= 128.
        im2 -= 1.
        return im2
        
    def pre_process(self, imgs):
        images_pre_processed = []
        
        for a_image in imgs:
            # a_image is of shape (dim1, dim2, channels)
            a_image = np.expand_dims(a_image, axis=0)  # line reshapes to (1, dim1, dim2, channels)
            images_pre_processed.append(self.preprocess_input(a_image))  # Preprocess the input image as per inceptionV3
        
        print("Length of np.vstack(images_pre_processed)..", len(np.vstack(images_pre_processed)))
        return np.vstack(images_pre_processed)  # Stack the images one below the other (rows)
		
	def rev_preprocess(self, im):
        im += 1
        im *= 128
        return im

    def predict(self, images):  # images will have values from 0 - 255
        """
        param images: a list of image tensors of shape (nSamples, H, W, C) ; 
		where H represents height, W represents width and C represents channels of an image respectively
        """
        probs = self.model.outputs[0]
        inputs = self.model.inputs[0]
        pre_processed_images = self.pre_process(images)

        return self.sess.run(probs, {inputs:pre_processed_images})

    def load_model(self, model_path, model_file_name):

        sess = tf.Session(graph=K.get_session().graph)
        K.set_session(sess)
        model = load_model(os.path.join(model_path, model_file_name))
        logits = model.outputs[0].op.inputs[0]
            
        print("Model loaded..")
        return sess, model, logits

PATH_TO_THE_MODEL = r''  # Absolute path to model
IMAGE_SAMPLES_FOLDER = r''  # Absolute path to folder where test data/Image samples are present
IMAGE_VS_LABELS_CSV = r''  # Absolute path to image-label csv
PATH_TO_SAVE_RESULTS = r''  # Absolute path to save results - must be <tomcat webapps folder>\teachntest\assets\results
PATH_TO_JSON_INDEX_CLASS_MAPPING = r''  # a local absolute path to a .json file that contains the index-class mapping
PROJECT_NAME = r''  # A string which represents a name under which an test/test is performed
model_file_name = r''  # A string which is the file name along with the extension of the model under test. A .h5 file in this case.

model = Model(PATH_TO_THE_MODEL, model_file_name)
print(model.logits)

test0 = AdversarialPatches( model,IMAGE_SAMPLES_FOLDER, IMAGE_VS_LABELS_CSV, PATH_TO_SAVE_RESULTS, PROJECT_NAME,
                            PATH_TO_JSON_INDEX_CLASS_MAPPING, 4, PATH_TO_THE_MODEL, model_file_name, 'input_1:0',
                            'predictions/BiasAdd:0', 16, (-1,1), 'circle', 8, learning_rate=5.0)
result = test0.run()
print(result)							
test1 = AdversarialInputs(model, IMAGE_SAMPLES_FOLDER, IMAGE_VS_LABELS_CSV, PATH_TO_SAVE_RESULTS, PROJECT_NAME, PATH_TO_JSON_INDEX_CLASS_MAPPING, 4, threshold=0.1)

result = test1.fgsm(0.015)  # epsilon value
print(result)

result = test1.cw() # Carlini & Wagner Method: can take optional learning rate & number of iterations
print(result)

test2 = FourierRadialFiltering(model, IMAGE_SAMPLES_FOLDER, IMAGE_VS_LABELS_CSV, PATH_TO_SAVE_RESULTS, PROJECT_NAME, PATH_TO_JSON_INDEX_CLASS_MAPPING, radius=0.5, threshold=0.1)
#
result = test2.run()
print(result)
#
test3 = ModelInterpretation(model, IMAGE_SAMPLES_FOLDER, IMAGE_VS_LABELS_CSV, PATH_TO_SAVE_RESULTS, PROJECT_NAME, PATH_TO_JSON_INDEX_CLASS_MAPPING, num_features=5, num_samples=10, hide_rest=True)
result = test3.run()
print(result)

test4 = ObjectRecognitionWeakSignals(model, IMAGE_SAMPLES_FOLDER, IMAGE_VS_LABELS_CSV, PATH_TO_SAVE_RESULTS, PROJECT_NAME, PATH_TO_JSON_INDEX_CLASS_MAPPING, threshold=0.1)
##
result = test4.generate_gray_scale()
print(result)
result = test4.generate_low_contrast(contrast_level_1=0.7)
print(result)
result = test4.generate_noisy(noise_width = 0.1, contrast_level_2 = 0.4)
print(result)
result = test4.generate_eidolon(grain = 10.0, coherence = 1.0, reach = 2.0)
print(result)
#
test5 = RotationTranslation(model, IMAGE_SAMPLES_FOLDER, IMAGE_VS_LABELS_CSV, PATH_TO_SAVE_RESULTS, PROJECT_NAME, PATH_TO_JSON_INDEX_CLASS_MAPPING, threshold=0.1)
result = test5.run()
print(result)

# PerformanceMetrics
pm = PerformanceMetrics(model, "input_1:0", PATH_TO_THE_MODEL, model_file_name, True, PATH_TO_SAVE_RESULTS,
                        PROJECT_NAME, IMAGE_SAMPLES_FOLDER, IMAGE_VS_LABELS_CSV, model.image_size_height,
                        model.image_size_width)
pm.compute()