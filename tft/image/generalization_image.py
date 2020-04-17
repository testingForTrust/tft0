# authors: TFTAuthors@accenture.com
# Purpose: This module abstracts different methods to audit an image classifier model for its ability to generalize

import os
import csv
import numpy as np
from keras.preprocessing import image

# imports specific to FourierFiltering Method
from scipy.fftpack import fft2, fftshift, ifft2, ifftshift

from .utils_image import read_images, read_img_names_labels_csv, save_images, rot_trans_class_overall_metrics_generator
from .utils_image import read_orig_dim_images_from_list_of_numpy_arrays
from .utils_image import get_imgs_clsfied_corr, output_csv_generator_rot_and_trans
from .utils_image import class_metrics_overall_csv_generator, output_csv_generator, create_output

# imports specific to ObjectRecognitionWeakSignals
import tft.image.wrapper as wr
from skimage.color import rgb2gray

import math
import json
import tensorflow as tf

import logging


class FourierRadialFiltering:
    """ Create radial filtered images and make an audit against the model. This is adapted from the following paper
    Source: https://gist.github.com/vis-opt/7b229ed3a299f3d469b7e66e039107cd
    Paper: http://arxiv.org/abs/1711.11561
    """

    def __init__(self, model, path_to_org_imgs, img_label_csv_path, path_to_save_res, project_name,
                 class_index_json_path, radius=0.35, threshold=0.1):

        """
        :param model: An object of a model class created by the user. Expected to have the parameters "self.sess",
               "self.x"(where x is a input placeholder) and "self.logits", "self.image_size_height",
               "self.image_size_width" and "self.num_channels"
        :param path_to_org_imgs: a string - a local absolute path to original images
        :param img_label_csv_path: a string - a local absolute path to a csv file that contains mapping between image
               names and corresponding labels. The column names of the csv file must be “ImageName” and “Label”
               respectively.
        :param path_to_save_res: a string - a local absolute path to save results of the audit
        :param project_name: a string - represents a name under which an audit/test is performed
        :param class_index_json_path: a string - a local absolute path to a .json file that contains the index-class
               mapping [Example: https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json]
        :param radius: Radius to be used while filtering the image in frequency domain. By default 0.35 [Reason:
               From the paper it was found that for cifar10 dataset, the radius selected was 11 which is found to be 35%
               of height or width of cifar10 image (i.e. 32).]
        :param threshold: a float value between 0 and 1. If the difference between the % of original examples correctly
               classified and % of adversarial examples correctly classified is greater than threshold, the metric of
               adversarial examples will be highlighted in red color in UI to indicate the model does not meet desired
               robustness. By default 0.1.
        """

        self.model = model
        self.PATH_TO_ORG_IMAGES = path_to_org_imgs
        self.PATH_TO_SAVE_RESULTS = path_to_save_res
        self.PROJECT_NAME = project_name
        self.CLASS_INDEX_JSON_PATH = class_index_json_path
        self.radius = radius
        self.threshold = threshold
        self.IMAGE_LABELS_CSV_PATH = img_label_csv_path
        self.BATCH_SIZE = 32  # For processing test data in batches of 32 samples

        # Generate experiments.csv.
        self.experiments_filepath = os.path.join(self.PATH_TO_SAVE_RESULTS, self.PROJECT_NAME)

        # A string value used to generate experiments.csv when user runs fourierfilter method
        self.method_type = "fourierfilteringValue"

        # Code added to generate the settings.csv file

        root_folder_to_save_settings = os.path.join(self.PATH_TO_SAVE_RESULTS, self.PROJECT_NAME, 'Generalization',
                                                    'FourierRadialFiltering')
        if not os.path.isdir(root_folder_to_save_settings):
            os.makedirs(root_folder_to_save_settings, exist_ok=True)
        # settings.csv
        with open(os.path.join(root_folder_to_save_settings, "settings.csv"), 'w', newline='') as output_file:
            output_writer = csv.writer(output_file, delimiter=',')
            output_writer.writerow(['Radius', 'Threshold'])
            output_writer.writerow([self.radius, self.threshold])
            output_writer.writerow(["a float (By default 0.35). Radius to be used while filtering the image in "
                                    "frequency domain.",
                                    "By default 0.1. [a float value between 0 and 1. The metric of perturbed examples"
                                    " will be highlighted in red color to indicate the model does not meet desired "
                                    "generalization ability if the difference between the % of original examples "
                                    "correctly classified and % of perturbed examples correctly classified is greater"
                                    " than threshold.]"])
        print("settings.csv has been generated..")

    @staticmethod
    def radial_filtering(input_image, rad):
        """
         Method that evaluates fourier filtered image using radial filtering method as described in the paper.
        :param input_image: numpy array of images.
        :param rad: radius to be used while filtering the image in frequency domain
        :return: an original image (numpy array) and fourier filtered image (numpy array) Shape: (h, w, c) [which will
                 be of actual original image dimensions]
        """
        perturbed_images = []
        for img in input_image:
            h, w, c = img.shape
            x = img.reshape(c, h, w)  # The reason we reshape is for the FFT and IFFT methods which demand that way
            # In general, if the image data is (H,W), the center is (H/2, W/2). This is required for the FFT shift.
            center_h, center_w = h / 2, w / 2
            # Create a Fourier mask:
            fft_mask = np.ones((c, h, w))
            for i in range(c):
                for j in range(h):
                    for k in range(w):
                        # from the paper it was found that for cifar10dataset, the radius selected was 11 which is found
                        # to be 35% of height or width (whichever is max).
                        if np.sqrt((j - center_h) ** 2 + (k - center_w) ** 2) >= (rad * h if h > w else rad * w):
                            fft_mask[i, j, k] = 0  # Default value is set to 0.
            # This is the radially filtered mask.
            # Compute the FFTs:
            x_fft = fftshift(fft2(x))
            # Mask the FFT:
            x_fft *= fft_mask
            # IFFT
            x_ifft = ifft2(ifftshift(x_fft)).real
            x_ifft = x_ifft.reshape(h, w, c)  # Back to original image's shape and range from (0-255); type float
            perturbed_images.append(x_ifft)
        return perturbed_images

    def run(self):
        """Main function"""
        # image_names: List of image names like a.JPEG, b.JPEG etc..
        # y : List of true label names for images
        image_names, y = read_img_names_labels_csv(self.IMAGE_LABELS_CSV_PATH)
        # For Example, if PATH_TO_SAVE_RESULTS is "C:\", then root_folder_to_save_images will be
        # "C:\Robustness\AttributeVariation\FourierRadialFiltering\Results"
        root_folder_to_save_images = os.path.join(self.PATH_TO_SAVE_RESULTS, self.PROJECT_NAME, 'Generalization',
                                                  'FourierRadialFiltering', 'Results')

        if not os.path.isdir(root_folder_to_save_images):
            os.makedirs(root_folder_to_save_images, exist_ok=True)

        # Method1:
        # First get all correctly classified data from whole testdata and then generate adv examples
        img_names_correctly_classified, y_, overall_testdata_pred_probs, org_imag_pred_prob = get_imgs_clsfied_corr(
            self.model,
            self.PATH_TO_ORG_IMAGES, image_names,
            self.model.image_size_height,
            self.model.image_size_width, y,
            self.CLASS_INDEX_JSON_PATH)
        samples_path_list = [os.path.join(self.PATH_TO_ORG_IMAGES, img_name) for img_name in
                             img_names_correctly_classified]

        input_size = len(samples_path_list)
        # batch_evaluator ==> Calculate number of iterations for batch_size; For example: if batch_size is 200 and if we
        # have 1000 images, then batch_evaluator is 5. And for 50K images batch_evaluator is 250
        batch_evaluator: int = math.ceil((input_size / self.BATCH_SIZE))  # Adding tye hint for the variable
        create_output(root_folder_to_save_images)

        with open(self.CLASS_INDEX_JSON_PATH) as index_class_mapping_file:
            mapping_dictionary = json.load(index_class_mapping_file)

        for i in range(batch_evaluator):
            index_from = self.BATCH_SIZE * i
            # ending index of the batch
            index_to = input_size if batch_evaluator - i == 1 else (i * self.BATCH_SIZE + self.BATCH_SIZE)
            image_names_batch = img_names_correctly_classified[index_from:index_to]

            y_batch = y_[index_from:index_to]
            # Get unique class names from the user supplied labels
            unique_label_names = list(set(y_batch))
            samples_ndarray_batch = read_images(samples_path_list[index_from:index_to],
                                                self.model.image_size_height,
                                                self.model.image_size_width)

            filtered_images_list = self.radial_filtering(samples_ndarray_batch, self.radius)
            stacked_filtered_images = read_orig_dim_images_from_list_of_numpy_arrays(filtered_images_list,
                                                                                     self.model.image_size_height,
                                                                                     self.model.image_size_width)

            save_images(unique_label_names, root_folder_to_save_images, image_names_batch, y_batch,
                        samples_ndarray_batch, stacked_filtered_images)

            # Make Predictions On Correctly classified Original And Perturbed Images
            # original_img_pred_probs_batch = self.model.predict(samples_ndarray_batch)
            original_img_pred_probs_batch = org_imag_pred_prob[index_from:index_to]
            perturbed_img_pred_probs_batch = self.model.predict(stacked_filtered_images)
            output_csv_generator(original_img_pred_probs_batch, perturbed_img_pred_probs_batch, mapping_dictionary,
                                 y_batch, image_names_batch, root_folder_to_save_images)
        print("output.csv has been generated..")
        return class_metrics_overall_csv_generator(overall_testdata_pred_probs, mapping_dictionary,
                                                   self.IMAGE_LABELS_CSV_PATH,
                                                   root_folder_to_save_images,
                                                   self.threshold, self.method_type, self.experiments_filepath)


class ObjectRecognitionWeakSignals:
    """
    Generates the manipulated images (low contrast, noisy, grey scale and eidolon) against the original image.
    Source: https://github.com/rgeirhos/object-recognition
    Paper: https://arxiv.org/abs/1706.06969
    """

    def __init__(self, model, path_to_org_imgs, img_label_csv_path, path_to_save_res, project_name,
                 class_index_json_path, threshold=0.1):

        # Please refer to the FourierFiltering class above for description of the below parameters.
        """
        :param model: An object of a model class created by the user. Expected to have the parameters "self.sess",
                "self.x"(where x is a input placeholder) and "self.logits", "self.image_size_height",
                "self.image_size_width" and "self.num_channels".
        :param path_to_org_imgs: Path to input image sample folder.
        :param img_label_csv_path: A string - a local absolute path to a csv file that contains mapping between image
               names and corresponding labels. The column names of the csv file must be “ImageName” and “Label”
               respectively.
        :param path_to_save_res: A string - a local absolute path to save results of the audit.
        :param project_name: A string - represents a name under which an audit/test is performed.
        :param class_index_json_path: A string - A local absolute path to a .json file that contains the index-class
               mapping [Example: https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json].
        :param threshold: A float value between 0 and 1. If the difference between the % of original examples correctly
               classified and % of adversarial examples correctly classified is greater than threshold, the metric of
               adversarial examples will be highlighted in red color in UI to indicate the model does not meet desired
               robustness. By default 0.1.
        """

        self.model = model
        self.PATH_TO_ORG_IMAGES = path_to_org_imgs
        self.PATH_TO_SAVE_RESULTS = path_to_save_res
        self.PROJECT_NAME = project_name
        self.IMAGE_LABELS_CSV_PATH = img_label_csv_path
        self.CLASS_INDEX_JSON_PATH = class_index_json_path
        self.threshold = threshold
        self.BATCH_SIZE = 32  # For processing test data in batches of 32 samples

        # Generate experiments.csv
        self.experiments_filepath = os.path.join(self.PATH_TO_SAVE_RESULTS, self.PROJECT_NAME)
        with open(self.CLASS_INDEX_JSON_PATH) as index_class_mapping_file:
            self.mapping_dictionary = json.load(index_class_mapping_file)

    @staticmethod
    def imload_rgb(img):
        """Load and return an RGB image in the range [0, 1]."""
        return img / 255.0

    @staticmethod
    def adjust_contrast(img, contrast_level):
        """Return the img scaled to a certain contrast level in [0, 1].
        parameters:
        - img: a numpy.ndarray
        - contrast_level: a scalar in [0, 1]; with 1 -> full contrast
        """
        assert (contrast_level >= 0.0), "contrast_level too low."
        assert (contrast_level <= 1.0), "contrast_level too high."
        return (1 - contrast_level) / 2.0 + img.dot(contrast_level)

    def grayscale_contrast(self, img, contrast_level):
        """Convert to grayscale. Adjust contrast.
        parameters:
        - img: a numpy.ndarray
        - contrast_level: a scalar in [0, 1]; with 1 -> full contrast
        """
        return self.adjust_contrast(img, contrast_level)

    def uniform_noise(self, img, width, contrast_level, rng):
        """Convert to grayscale. Adjust contrast. Apply uniform noise.
        parameters:
        - img: a numpy.ndarray
        - width: a scalar indicating width of additive uniform noise
                 -> then noise will be in range [-width, width]
        - contrast_level: a scalar in [0, 1]; with 1 -> full contrast
        - rng: a np.random.RandomState(seed=XYZ) to make it reproducible
        """

        img = self.grayscale_contrast(img, contrast_level)
        return self.apply_uniform_noise(img, -width, width, rng)

    def apply_uniform_noise(self, img, low, high, rng=None):
        """Apply uniform noise to an img, clip outside values to 0 and 1.
        parameters:
        - img: a numpy.ndarray
        - low: lower bound of noise within [low, high)
        - high: upper bound of noise within [low, high)
        - rng: a np.random.RandomState(seed=XYZ) to make it reproducible
        """
        nrow = img.shape[0]
        ncol = img.shape[1]
        noise = self.get_uniform_noise(low, high, nrow, ncol, rng)
        img = img + noise.reshape(nrow, ncol, 1)
        # clip values
        img = np.where(img < 0, 0, img)
        img = np.where(img > 1, 1, img)

        assert self.is_in_bounds(img, 0, 1), "values <0 or >1 occurred"
        return img

    @staticmethod
    def get_uniform_noise(low, high, nrow, ncol, rng=None):
        """Return uniform noise within [low, high) of size (nrow, ncol).
        parameters:
        - low: lower bound of noise within [low, high)
        - high: upper bound of noise within [low, high)
        - nrow: number of rows of desired noise
        - ncol: number of columns of desired noise
        - rng: a np.random.RandomState(seed=XYZ) to make it reproducible
        """
        if rng is None:
            return np.random.uniform(low=low, high=high, size=(nrow, ncol))
        else:
            return rng.uniform(low=low, high=high, size=(nrow, ncol))

    @staticmethod
    def is_in_bounds(mat, low, high):
        """Return whether all values in 'mat' fall between low and high.
        parameters:
        - mat: a numpy.ndarray
        - low: lower bound (inclusive)
        - high: upper bound (inclusive)
        """
        return np.all(np.logical_and(mat >= low, mat <= high))

    @staticmethod
    def eidolon_partially_coherent_disarray(img_list, reach, coherence, grain):
        """Return parametrically distorted images (produced by Eidolon factory.
        For more information on the effect of different distortions, please
        have a look at the paper: Koenderink et al., JoV 2017,
        Eidolons: Novel stimuli for vision research).
        - img: a numpy.ndarray
        - reach: float, controlling the strength of the manipulation
        - coherence: a float within [0, 1] with 1 = full coherence
        - grain: float, controlling how fine-grained the distortion is
        """
        eidolon_images_list = []
        for img in img_list:
            img_eidolon = wr.partially_coherent_disarray(wr.data_to_pic(img), reach, coherence, grain)
            h, w = img_eidolon.shape
            reshaped_eidolon_img = img_eidolon.reshape(h, w, 1)
            # org_eidolon_img = np.concatenate((reshaped_eidolon_img, reshaped_eidolon_img, reshaped_eidolon_img),
            #                                 axis=2)
            eidolon_images_list.append(
                np.concatenate((reshaped_eidolon_img, reshaped_eidolon_img, reshaped_eidolon_img),
                               axis=2))
        return eidolon_images_list

    def get_gray_scale_image(self, image_arrays):
        # convert a PIL image to a np array. Handles grayscale images as well. dtype=float32
        org_gray_scale_img = []
        for img in image_arrays:
            original_image = image.img_to_array(img).astype(np.uint8)
            rgb_img = self.imload_rgb(original_image)  # rgb image - values between 0 - 1

            ###################################################
            # A) Example for color-experiment:
            #    - convert to grayscale
            ###################################################

            gray_scale_img = rgb2gray(rgb_img)
            h, w = gray_scale_img.shape
            temp_gray_img = gray_scale_img.reshape(h, w, 1)  # Reshape with channel = 1
            # stack 2D arrays (gray scale image) into a single 3D array for processing by model
            org_gray_scale_img.append(np.concatenate((temp_gray_img, temp_gray_img, temp_gray_img), axis=2))
        return org_gray_scale_img

    def generate_gray_scale(self):
        # image_names: List of image names like a.JPEG, b.JPEG etc..
        # y : List of true label names for images
        image_names, y = read_img_names_labels_csv(self.IMAGE_LABELS_CSV_PATH)
        # For Example, if PATH_TO_SAVE_RESULTS is "C:\", then root_folder_to_save_images will be
        # "C:\Robustness\AttributeVariation\FourierRadialFiltering\Results"

        method_type = "colortograyscaleValue"

        # Code added to generate the settings.csv file

        root_folder_to_save_settings = os.path.join(self.PATH_TO_SAVE_RESULTS, self.PROJECT_NAME, 'Generalization',
                                                    'Object_Recognition', 'GrayScale')
        if not os.path.isdir(root_folder_to_save_settings):
            os.makedirs(root_folder_to_save_settings, exist_ok=True)
        # settings.csv
        with open(os.path.join(root_folder_to_save_settings, "settings.csv"), 'w', newline='') as output_file:
            output_writer = csv.writer(output_file, delimiter=',')
            output_writer.writerow(['Threshold'])
            output_writer.writerow([self.threshold])
            output_writer.writerow(["By default 0.1. [a float value between 0 and 1. The metric of perturbed examples"
                                    " will be highlighted in red color to indicate the model does not meet desired"
                                    " generalization ability if the difference between the % of original examples "
                                    "correctly classified and % of perturbed examples correctly classified is greater"
                                    " than threshold.]"])
        print("grayscale_settings.csv has been generated..")

        root_folder_to_save_images = os.path.join(self.PATH_TO_SAVE_RESULTS, self.PROJECT_NAME, 'Generalization',
                                                  'Object_Recognition', 'GrayScale', 'Results')
        if not os.path.isdir(root_folder_to_save_images):
            os.makedirs(root_folder_to_save_images, exist_ok=True)

        img_names_correctly_classified, y_, overall_testdata_pred_probs, org_imag_pred_prob = \
            get_imgs_clsfied_corr(self.model, self.PATH_TO_ORG_IMAGES, image_names, self.model.image_size_height,
                                  self.model.image_size_width, y, self.CLASS_INDEX_JSON_PATH)

        samples_path_list = [os.path.join(self.PATH_TO_ORG_IMAGES, img_name)
                             for img_name in img_names_correctly_classified]

        input_size = len(samples_path_list)
        # batch_evaluator ==> Calculate number of iterations for batch_size; For example: if batch_size is 200
        # and if we have 1000 images, then batch_evaluator is 5. And for 50K images batch_evaluator is 250
        batch_evaluator: int = math.ceil((input_size / self.BATCH_SIZE))
        create_output(root_folder_to_save_images)
        for i in range(batch_evaluator):
            # starting index of the batch
            index_from = self.BATCH_SIZE * i
            # ending index of the batch
            index_to = input_size if batch_evaluator - i == 1 else (i * self.BATCH_SIZE + self.BATCH_SIZE)
            image_names_batch = img_names_correctly_classified[index_from:index_to]

            y_batch = y_[index_from:index_to]
            # Get unique class names from the user supplied labels
            unique_label_names = list(set(y_batch))
            samples_ndarray_batch = read_images(samples_path_list[index_from:index_to],
                                                self.model.image_size_height, self.model.image_size_width)
            gray_scale_images_list = self.get_gray_scale_image(samples_ndarray_batch)
            stacked_gray_scale_images = read_orig_dim_images_from_list_of_numpy_arrays(gray_scale_images_list,
                                                                                       self.model.image_size_height,
                                                                                       self.model.image_size_width)

            save_images(unique_label_names, root_folder_to_save_images, image_names_batch, y_batch,
                        samples_ndarray_batch, stacked_gray_scale_images)

            # Make Predictions On Correctly classified Original And Adversarial Images
            # original_img_pred_probs = self.model.predict(samples_ndarray_batch)
            original_img_pred_probs = org_imag_pred_prob[index_from:index_to]
            perturbed_img_pred_probs = self.model.predict(stacked_gray_scale_images)
            output_csv_generator(original_img_pred_probs, perturbed_img_pred_probs, self.mapping_dictionary,
                                 y_batch, image_names_batch, root_folder_to_save_images)
        return class_metrics_overall_csv_generator(overall_testdata_pred_probs, self.mapping_dictionary,
                                                   self.IMAGE_LABELS_CSV_PATH,
                                                   root_folder_to_save_images,
                                                   self.threshold, method_type, self.experiments_filepath)

    def generate_low_contrast(self, contrast_level_1=0.1):
        # image_names: List of image names like a.JPEG, b.JPEG etc..
        # y : List of true label names for images
        image_names, y = read_img_names_labels_csv(self.IMAGE_LABELS_CSV_PATH)
        # For Example, if PATH_TO_SAVE_RESULTS is "C:\", then root_folder_to_save_images will be
        # "C:\Robustness\AttributeVariation\FourierRadialFiltering\Results"

        method_type = "contrastValue"

        root_folder_to_save_settings = os.path.join(self.PATH_TO_SAVE_RESULTS, self.PROJECT_NAME, 'Generalization',
                                                    'Object_Recognition', 'LowContrast')
        if not os.path.isdir(root_folder_to_save_settings):
            os.makedirs(root_folder_to_save_settings, exist_ok=True)
        # settings.csv
        with open(os.path.join(root_folder_to_save_settings, "settings.csv"), 'w', newline='') as \
                output_file:
            output_writer = csv.writer(output_file, delimiter=',')
            output_writer.writerow(['Threshold', 'Contrast Level'])
            output_writer.writerow([self.threshold, contrast_level_1])
            output_writer.writerow(["By default 0.1. [a float value between 0 and 1. The metric of noisy examples "
                                    "will be highlighted in red color to indicate the model does not meet desired "
                                    "generalization ability if the difference between the % of original examples "
                                    "correctly classified and % of noisy examples correctly classified is greater "
                                    "than threshold.]",
                                    "Contrast Level (a float between 0 to 1. By default 0.1) that indicates by what "
                                    "fraction the original images were perturbed with respect to contrast "
                                    "of the image"])
        print("lowcontrast_settings.csv has been generated..")

        root_folder_to_save_images = os.path.join(self.PATH_TO_SAVE_RESULTS, self.PROJECT_NAME, 'Generalization',
                                                  'Object_Recognition', 'LowContrast', 'Results')

        if not os.path.isdir(root_folder_to_save_images):
            os.makedirs(root_folder_to_save_images, exist_ok=True)

        img_names_correctly_classified, y_, overall_testdata_pred_probs, org_imag_pred_prob = \
            get_imgs_clsfied_corr(self.model, self.PATH_TO_ORG_IMAGES, image_names, self.model.image_size_height,
                                  self.model.image_size_width, y, self.CLASS_INDEX_JSON_PATH)
        samples_path_list = [os.path.join(self.PATH_TO_ORG_IMAGES, img_name)
                             for img_name in img_names_correctly_classified]

        input_size = len(samples_path_list)
        # batch_evaluator ==> Calculate number of iterations for batch_size; For example: if batch_size is 200
        # and if we have 1000 images, then batch_evaluator is 5. And for 50K images batch_evaluator is 250
        batch_evaluator: int = math.ceil((input_size / self.BATCH_SIZE))
        create_output(root_folder_to_save_images)
        for i in range(batch_evaluator):
            # starting index of the batch
            index_from = self.BATCH_SIZE * i
            # ending index of the batch
            index_to = input_size if batch_evaluator - i == 1 else (i * self.BATCH_SIZE + self.BATCH_SIZE)
            image_names_batch = img_names_correctly_classified[index_from:index_to]

            y_batch = y_[index_from:index_to]
            # Get unique class names from the user supplied labels in a batch
            unique_label_names = list(set(y_batch))
            samples_ndarray_batch = read_images(samples_path_list[index_from:index_to],
                                                self.model.image_size_height, self.model.image_size_width)
            # for img_path in samples_path_list[index_from:index_to]:
            # img = image.load_img(img_path)  # Read image in a PIL format
            # convert a PIL image to a np array. Handles grayscale images as well. dtype=float32
            # original_image = image.img_to_array(img).astype(np.uint8)
            low_contrast_images_list = []
            for img in samples_ndarray_batch:
                low_contrast_images_list.append(self.grayscale_contrast(img / 255, contrast_level=contrast_level_1))

            stacked_low_contrast_images = np.vstack([np.expand_dims(x, axis=0) for x in low_contrast_images_list])
            save_images(unique_label_names, root_folder_to_save_images, image_names_batch,
                        y_batch, samples_ndarray_batch, stacked_low_contrast_images)

            # Make Predictions On Correctly classified Original And Adversarial Images
            # original_img_pred_probs = self.model.predict(samples_ndarray_batch)
            original_img_pred_probs = org_imag_pred_prob[index_from:index_to]
            # multiplying by 255.0 since pre_processing would be part of predict method in model class
            perturbed_img_pred_probs = self.model.predict(stacked_low_contrast_images*255.)

            output_csv_generator(original_img_pred_probs, perturbed_img_pred_probs, self.mapping_dictionary,
                                 y_batch, image_names_batch, root_folder_to_save_images)

        return class_metrics_overall_csv_generator(overall_testdata_pred_probs, self.mapping_dictionary,
                                                   self.IMAGE_LABELS_CSV_PATH,
                                                   root_folder_to_save_images,
                                                   self.threshold, method_type, self.experiments_filepath)

    def generate_noisy(self, noise_width=0.1, contrast_level_2=0.3):
        rng = np.random.RandomState(seed=42)
        # image_names: List of image names like a.JPEG, b.JPEG etc..
        # y : List of true label names for images
        image_names, y = read_img_names_labels_csv(self.IMAGE_LABELS_CSV_PATH)
        # For Example, if PATH_TO_SAVE_RESULTS is "C:\", then root_folder_to_save_images will be
        # "C:\Robustness\AttributeVariation\FourierRadialFiltering\Results"

        method_type = "additivenoiseValue"

        # Code added to generate the settings.csv file

        root_folder_to_save_settings = os.path.join(self.PATH_TO_SAVE_RESULTS, self.PROJECT_NAME, 'Generalization',
                                                    'Object_Recognition', 'Noisy')
        if not os.path.isdir(root_folder_to_save_settings):
            os.makedirs(root_folder_to_save_settings, exist_ok=True)
        # settings.csv
        with open(os.path.join(root_folder_to_save_settings, "settings.csv"), 'w', newline='') as output_file:
            output_writer = csv.writer(output_file, delimiter=',')
            output_writer.writerow(['Threshold', 'Noise Width', 'Contrast Level'])
            output_writer.writerow([self.threshold, noise_width, contrast_level_2])
            output_writer.writerow(["By default 0.1. [a float value between 0 and 1. The metric of noisy examples "
                                    "will be highlighted in red color to indicate the model does not meet desired "
                                    "generalization ability if the difference between the % of original examples "
                                    "correctly classified and % of noisy examples correctly classified is greater "
                                    "than threshold.]",
                                    "a float value (between 0 to 1). Indicates the fraction of noise to be introduced",
                                    "Contrast Level (a float between 0 to 1. By default 0.1) that indicates by what "
                                    "fraction the original images were perturbed with respect to contrast "
                                    "of the image"])
        print("additive_settings.csv has been generated..")

        root_folder_to_save_images = os.path.join(self.PATH_TO_SAVE_RESULTS, self.PROJECT_NAME, 'Generalization',
                                                  'Object_Recognition', 'Noisy', 'Results')
        if not os.path.isdir(root_folder_to_save_images):
            os.makedirs(root_folder_to_save_images, exist_ok=True)
        img_names_correctly_classified, y_, overall_testdata_pred_probs, org_imag_pred_prob = \
            get_imgs_clsfied_corr(self.model, self.PATH_TO_ORG_IMAGES, image_names, self.model.image_size_height,
                                  self.model.image_size_width, y, self.CLASS_INDEX_JSON_PATH)

        samples_path_list = [os.path.join(self.PATH_TO_ORG_IMAGES, img_name)
                             for img_name in img_names_correctly_classified]

        input_size = len(samples_path_list)
        # batch_evaluator ==> Calculate number of iterations for batch_size; For example: if batch_size is 200
        # and if we have 1000 images, then batch_evaluator is 5. And for 50K images batch_evaluator is 250
        batch_evaluator: int = math.ceil((input_size / self.BATCH_SIZE))
        create_output(root_folder_to_save_images)
        for i in range(batch_evaluator):
            # starting index of the batch
            index_from = self.BATCH_SIZE * i
            # ending index of the batch
            index_to = input_size if batch_evaluator - i == 1 else (i * self.BATCH_SIZE + self.BATCH_SIZE)
            image_names_batch = img_names_correctly_classified[index_from:index_to]

            y_batch = y_[index_from:index_to]
            # Get unique class names from the user supplied labels
            unique_label_names = list(set(y_batch))
            samples_ndarray_batch = read_images(samples_path_list[index_from:index_to],
                                                self.model.image_size_height, self.model.image_size_width)

            # original_images_list = []
            noisy_images_list = []
            for img in samples_ndarray_batch:
                # convert a PIL image to a np array. Handles grayscale images as well. dtype=float32
                original_image = image.img_to_array(img).astype(np.uint8)
                noisy_img = self.uniform_noise(img=self.imload_rgb(original_image), width=noise_width,
                                               contrast_level=contrast_level_2,
                                               rng=rng)
                noisy_images_list.append(noisy_img)
            stacked_noisy_images = read_orig_dim_images_from_list_of_numpy_arrays(noisy_images_list,
                                                                                  self.model.image_size_height,
                                                                                  self.model.image_size_width)

            save_images(unique_label_names, root_folder_to_save_images, image_names_batch, y_batch,
                        samples_ndarray_batch, stacked_noisy_images)

            #  Make Predictions On Correctly classified Original And Perturbed Images
            #  original_img_pred_probs = self.model.predict(samples_ndarray_batch)
            original_img_pred_probs = org_imag_pred_prob[index_from:index_to]
            perturbed_img_pred_probs = self.model.predict(stacked_noisy_images)

            output_csv_generator(original_img_pred_probs, perturbed_img_pred_probs, self.mapping_dictionary,
                                 y_batch, image_names_batch, root_folder_to_save_images)
        return class_metrics_overall_csv_generator(overall_testdata_pred_probs, self.mapping_dictionary,
                                                   self.IMAGE_LABELS_CSV_PATH,
                                                   root_folder_to_save_images,
                                                   self.threshold, method_type, self.experiments_filepath)

    def generate_eidolon(self, grain=10.0, coherence=1.0, reach=8.0):
        # image_names: List of image names like a.JPEG, b.JPEG etc..
        # y : List of true label names for images
        image_names, y = read_img_names_labels_csv(self.IMAGE_LABELS_CSV_PATH)
        # For Example, if PATH_TO_SAVE_RESULTS is "C:\", then root_folder_to_save_images will be
        # "C:\Robustness\AttributeVariation\FourierRadialFiltering\Results"
        method_type = "eidolonnoiseValue"

        root_folder_to_save_settings = os.path.join(self.PATH_TO_SAVE_RESULTS, self.PROJECT_NAME, 'Generalization',
                                                    'Object_Recognition', 'Eidolon')
        if not os.path.isdir(root_folder_to_save_settings):
            os.makedirs(root_folder_to_save_settings, exist_ok=True)
        # settings.csv
        with open(os.path.join(root_folder_to_save_settings, "settings.csv"), 'w', newline='') as output_file:
            output_writer = csv.writer(output_file, delimiter=',')
            output_writer.writerow(['Grain', 'Coherence', 'Reach', 'Threshold'])
            output_writer.writerow([grain, coherence, reach, self.threshold])
            output_writer.writerow(["a float (by default 10.0). Controls how fine-grained the distortion is",
                                    "a float within 0 to 1. By default 1.0 with 1 ==> full coherence",
                                    "a float (by default 8.0). Controls the strength of the manipulation.",
                                    "By default 0.1.  [a float value between 0 and 1. The metric of perturbed examples"
                                    " will be highlighted in red color to indicate the model does not meet desired "
                                    "generalization ability if the difference between the % of original examples "
                                    "correctly classified and % of perturbed examples correctly classified is greater "
                                    "than threshold.]"])
        print("eidolon_settings.csv has been generated..")

        root_folder_to_save_images = os.path.join(self.PATH_TO_SAVE_RESULTS, self.PROJECT_NAME, 'Generalization',
                                                  'Object_Recognition', 'Eidolon', 'Results')
        if not os.path.isdir(root_folder_to_save_images):
            os.makedirs(root_folder_to_save_images, exist_ok=True)
        img_names_correctly_classified, y_, overall_testdata_pred_probs, org_imag_pred_prob = \
            get_imgs_clsfied_corr(self.model, self.PATH_TO_ORG_IMAGES, image_names, self.model.image_size_height,
                                  self.model.image_size_width, y, self.CLASS_INDEX_JSON_PATH)

        samples_path_list = [os.path.join(self.PATH_TO_ORG_IMAGES, img_name)
                             for img_name in img_names_correctly_classified]

        input_size = len(samples_path_list)
        # batch_evaluator ==> Calculate number of iterations for batch_size; For example: if batch_size is 200
        # and if we have 1000 images, then batch_evaluator is 5. And for 50K images batch_evaluator is 250
        batch_evaluator: int = math.ceil((input_size / self.BATCH_SIZE))
        create_output(root_folder_to_save_images)
        for i in range(batch_evaluator):
            # starting index of the batch
            index_from = self.BATCH_SIZE * i
            # ending index of the batch
            index_to = input_size if batch_evaluator - i == 1 else (i * self.BATCH_SIZE + self.BATCH_SIZE)
            image_names_batch = img_names_correctly_classified[index_from:index_to]

            y_batch = y_[index_from:index_to]
            # Get unique class names from the user supplied labels
            unique_label_names = list(set(y_batch))
            samples_ndarray_batch = read_images(samples_path_list[index_from:index_to],
                                                self.model.image_size_height, self.model.image_size_width)
            eidolon_images_list = self.eidolon_partially_coherent_disarray(samples_ndarray_batch, reach, coherence,
                                                                           grain)
            stacked_eidolon_images = read_orig_dim_images_from_list_of_numpy_arrays(eidolon_images_list,
                                                                                    self.model.image_size_height,
                                                                                    self.model.image_size_width)

            save_images(unique_label_names, root_folder_to_save_images, image_names_batch, y_batch,
                        samples_ndarray_batch, stacked_eidolon_images)

            # Make Predictions On Correctly classified Original And Perturbed Images
            # original_img_pred_probs = self.model.predict(samples_ndarray_batch)
            original_img_pred_probs = org_imag_pred_prob[index_from:index_to]
            perturbed_img_pred_probs = self.model.predict(stacked_eidolon_images)

            output_csv_generator(original_img_pred_probs, perturbed_img_pred_probs, self.mapping_dictionary,
                                 y_batch, image_names_batch, root_folder_to_save_images)

        return class_metrics_overall_csv_generator(overall_testdata_pred_probs, self.mapping_dictionary,
                                                   self.IMAGE_LABELS_CSV_PATH,
                                                   root_folder_to_save_images,
                                                   self.threshold, method_type, self.experiments_filepath)


class RotationTranslation:
    """
    This class abstracts the idea to fool CNNs with Simple Transformations like rotating and translating images
    Paper: https://arxiv.org/pdf/1712.02779
    Title: A Rotation and a Translation Suffice: Fooling CNNs with Simple Transformations
    """

    def __init__(self, model, path_to_org_imgs, img_label_csv_path, path_to_save_res, project_name,
                 class_index_json_path, threshold=0.1):
        """
        :param model: An object of a model class created by the user. Expected to have the parameters "self.sess",
                "self.x"(where x is a input placeholder) and "self.logits", "self.image_size_height",
                "self.image_size_width" and "self.num_channels".
        :param path_to_org_imgs: Path to input image sample folder.
        :param img_label_csv_path: A string - a local absolute path to a csv file that contains mapping between image
               names and corresponding labels. The column names of the csv file must be “ImageName” and “Label”
               respectively.
        :param path_to_save_res: A string - a local absolute path to save results of the audit.
        :param project_name: A string - represents a name under which an audit/test is performed.
        :param class_index_json_path: A string - A local absolute path to a .json file that contains the index-class
               mapping [Example: https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json].
        :param threshold: A float value between 0 and 1. If the difference between the % of original examples correctly
               classified and % of adversarial examples correctly classified is greater than threshold, the metric of
               adversarial examples will be highlighted in red color in UI to indicate the model does not meet desired
               robustness. By default 0.1.
        """
        self.model = model
        self.PATH_TO_ORG_IMAGES = path_to_org_imgs
        self.IMAGE_LABELS_CSV_PATH = img_label_csv_path
        self.PATH_TO_SAVE_RESULTS = path_to_save_res
        self.PROJECT_NAME = project_name
        self.CLASS_INDEX_JSON_PATH = class_index_json_path
        self.threshold = threshold
        self.BATCH_SIZE = 32
        self.method_type = "rotationtranslationValue"
        self.experiments_filepath = os.path.join(self.PATH_TO_SAVE_RESULTS, self.PROJECT_NAME)

        # Code added to generate the settings.csv file

        root_folder_to_save_settings = os.path.join(self.PATH_TO_SAVE_RESULTS, self.PROJECT_NAME, 'Generalization',
                                                    'RotationNTranslation')
        if not os.path.isdir(root_folder_to_save_settings):
            os.makedirs(root_folder_to_save_settings, exist_ok=True)
        # settings.csv
        with open(os.path.join(root_folder_to_save_settings, "settings.csv"), 'w', newline='') as output_file:
            output_writer = csv.writer(output_file, delimiter=',')
            output_writer.writerow(['Threshold'])
            output_writer.writerow([self.threshold])
            output_writer.writerow(["By default 0.1. [a float value between 0 and 1. The metric of perturbed examples"
                                    " will be highlighted in red color to indicate the model does not meet desired "
                                    "generalization ability if the difference between the % of original examples "
                                    "correctly classified and % of perturbed examples correctly classified is greater "
                                    "than threshold.]"])
        print("settings.csv has been generated..")

    def grid_search(self, batch_samples, batch_img_names, batch_labels, image_height, image_width,
                    mapping_dictionary, root_folder_to_save_images, orig_img_pred_probs):

        # In order to maintain the visual similarity of images to the natural ones the space of allowed perturbations
        # is restricted to be relatively small. 30% at most for rotations and 10% of image size in each direction for
        # translation. [As Per The Paper]
        """
        :param batch_samples: List of correctly classified images for one batch.
        :param batch_img_names: List of image names for one batch.
        :param batch_labels: List of image labels for one batch.
        :param image_height: Image height, expected as per the model.
        :param image_width: Image width, expected as per the model.
        :param mapping_dictionary: A dictionary mapped to the index json file of class labels with indices.
        :param root_folder_to_save_images: Path to save the results.
        :param orig_img_pred_probs: A List - contains the prediction for a batch of correctly classified images.
        """

        max_rotation_degrees = 30
        rotations = [i for i in range(1, max_rotation_degrees + 1)]  # 1 till 30 degrees
        max_translation_pixels = (round(0.1 * image_height), round(0.1 * image_width))
        # [1,1] till [max_translation_pixels,max_translation_pixels]
        translations = [[i, i] for i in range(1, max_translation_pixels[0] + 1)]

        adversarial_image_list = []  # initialize a list variable to hold different r&t variations under attack space
        adversarial_image_name_list = []  # a list of human readable names for adversarial examples
        orig_image_wrt_adv_image_list = []  # a list of ndarrays where adversarial image has been found

        y_ = []  # array of actual labels of the original images where adversarial have been found
        y_pred = []  # predictions on original image where adversarial have been found
        y_pred_adv = []  # predictions on adversarial image

        # Convert the images in the list to a 4D numpy array of shape (nSamples, h, w, c)
        batch_samples_4d_array = read_orig_dim_images_from_list_of_numpy_arrays(batch_samples,
                                                                                self.model.image_size_height,
                                                                                self.model.image_size_width)
        # Create two tensorflow placeholders for rotations and translations respectively
        rot_placeholder = tf.placeholder(dtype=np.uint8, shape=(None, self.model.image_size_height,
                                                                self.model.image_size_width, 3))
        trans_placeholder = tf.placeholder(dtype=np.uint8, shape=(None, self.model.image_size_height,
                                                                  self.model.image_size_width, 3))
        # Grid Search algorithm starts here.
        for theta in rotations:
            logging.info("Running For Theta '%s' degree/s" % theta)
            with tf.Session() as sess:
                sess.graph.as_default()  # to avoid possible mem leaks
                # angles to be provided in radians to this rotate function. hence theta is multiplied by pi/180.
                rotated_img_tensor = tf.contrib.image.rotate(rot_placeholder, theta * math.pi / 180)
                # Rotate all the images in the batch at once for a value of theta.
                # rot_images_batch is an ndarray of shape (nSamples, h, w, c)
                rot_images_batch = sess.run(rotated_img_tensor, feed_dict={rot_placeholder: batch_samples_4d_array})
                logging.info("rot_images_batch.shape " + str(rot_images_batch.shape))
            # Translate the rotated images in the batch
            for a_translation_dx_dy in translations:
                logging.info("Translating with pixels " + str(a_translation_dx_dy))
                global found  # a flag to identify if adv img is found or not found
                # global index  # index for getting adversarial img array if it is found (from rot_trans_batch)
                found = False
                # index = 0
                orig_img_names_wrt_adv = []  # a list indexes of original samples where adv img has been found
                with tf.Session() as sess:
                    sess.graph.as_default()  # to avoid possible mem leaks
                    rot_trans_tensor = tf.contrib.image.translate(trans_placeholder, a_translation_dx_dy)
                    # rot_trans_batch is an ndarray of shape (nSamples, h, w, c)
                    rot_trans_batch = sess.run(rot_trans_tensor, feed_dict={trans_placeholder: rot_images_batch})
                # get predictions on rot_trans_batch and original images. And then check for each variant if pred class
                # is not equal to label [in order to get adversarial example].
                # If adversarial is found for a variant of an image, then
                # [A] remove the image from rot_images_batch so that it is not considered for subsequent translations.
                # [B} remove the image related data from orig img list (i.e batch_samples_4d_array), batch_img_names &
                # batch_labels such that they are not considered in subsequent perturbations using theta
                perturbed_img_pred_probs = self.model.predict(rot_trans_batch)
                # orig_img_pred_probs = self.model.predict(batch_samples_4d_array)
                # orig_img_pred_probs contains the probability fo original images in a batch
                for preds, pred_orig, img_name, label in zip(perturbed_img_pred_probs, orig_img_pred_probs,
                                                             batch_img_names, batch_labels):
                    perturbed_top_predicted_label_index = preds.argsort()[-1]
                    predicted_class: str = mapping_dictionary[str(perturbed_top_predicted_label_index)]
                    org_image_top_predicted_label_index = pred_orig.argsort()[-1]
                    org_image_predicted_class = mapping_dictionary[str(org_image_top_predicted_label_index)]
                    # Check if pred class is not equal to label to get adversarial example
                    if predicted_class.lower() != label.lower():
                        index = batch_img_names.index(img_name)
                        adv_img_name = img_name.split('.')[0] + '_' + str(theta)\
                                       + '_' + str(a_translation_dx_dy[0]) \
                                       + '-' + str(a_translation_dx_dy[1]) + '.JPEG'
                        logging.info("Found a adversarial - " + adv_img_name)
                        orig_img_names_wrt_adv.append(index)
                        print(orig_img_names_wrt_adv)
                        found = True
                        adversarial_image_name_list.append(adv_img_name)
                        adversarial_image_list.append(rot_trans_batch[index])
                        orig_image_wrt_adv_image_list.append(batch_samples_4d_array[index])
                        y_.append(label)
                        y_pred.append(org_image_predicted_class)
                        y_pred_adv.append(predicted_class)
                    else:
                        found = False
                # delete those image related data where adversarial samples have been found such that they are not
                # considered for subsequent translations and rotations
                if len(orig_img_names_wrt_adv) != 0:
                    indexes_to_del = tuple(orig_img_names_wrt_adv)
                    logging.info("Deleting image related data where adversarial samples have been found such that they "
                                 "are not considered for subsequent translations and rotations")
                    rot_images_batch = np.delete(rot_images_batch, indexes_to_del, 0)
                    batch_samples_4d_array = np.delete(batch_samples_4d_array, indexes_to_del, 0)
                    logging.info("Shape of rot_img_batch and batch_samples after deleting" + str(rot_images_batch.shape)
                                 + str(batch_samples_4d_array.shape))
                    batch_img_names = list(np.delete(batch_img_names, indexes_to_del))
                    batch_labels = list(np.delete(batch_labels, indexes_to_del))
                    orig_img_pred_probs = np.delete(orig_img_pred_probs, indexes_to_del, 0)
                    # Break the inner loop if there is no image to rotate from the batch
                    if len(orig_img_pred_probs) == 0:
                        break
                # logging.info("Sample Names And Labels Considered For Subsequent Translations and Rotations" +
            # Break the inner loop if there is no image to rotate from the batch
            if len(orig_img_pred_probs) == 0:
                break

        if len(adversarial_image_name_list) > 0:
            save_images(list(set(y_)), root_folder_to_save_images, adversarial_image_name_list, y_,
                        orig_image_wrt_adv_image_list, adversarial_image_list)

        return adversarial_image_name_list, y_, y_pred, y_pred_adv

    def run(self):
        # image_names: List of image names like a.JPEG, b.JPEG etc..
        # y : List of true label names for images
        image_names, y = read_img_names_labels_csv(self.IMAGE_LABELS_CSV_PATH)
        img_height = self.model.image_size_height
        img_width = self.model.image_size_width
        #  correctly classified images from the test data
        img_names_correctly_classified, y_, overall_testdata_pred_probs, org_imag_pred_prob = \
            get_imgs_clsfied_corr(self.model, self.PATH_TO_ORG_IMAGES, image_names, img_height, img_width, y,
                                  self.CLASS_INDEX_JSON_PATH)
        samples_path_list = [os.path.join(self.PATH_TO_ORG_IMAGES, img_name)
                             for img_name in img_names_correctly_classified]

        # images are of different dimensions
        original_images_list = []
        for input_image_path in samples_path_list:
            img = image.load_img(input_image_path)  # Read image in a PIL format
            # convert a PIL image to a np array. Handles grayscale images as well. dtype=float32
            original_image = image.img_to_array(img).astype(np.uint8)
            original_images_list.append(original_image)

        input_size = len(samples_path_list)
        # Calculate number of iterations for BATCH_SIZE:for 1000 images, it is 5. For 50K images it is 250
        batch_evaluator: int = math.ceil((input_size / self.BATCH_SIZE))

        root_folder_to_save_img = os.path.join(self.PATH_TO_SAVE_RESULTS, self.PROJECT_NAME, 'Generalization',
                                               'RotationNTranslation', 'Results')
        if not os.path.isdir(root_folder_to_save_img):
            os.makedirs(root_folder_to_save_img, exist_ok=True)
        with open(self.CLASS_INDEX_JSON_PATH) as index_class_mapping_file:
            mapping_dict = json.load(index_class_mapping_file)

        y_truth = []  # list of truth values of test data where adversarial example has been found (for all batches)
        logging.basicConfig(filename='RnT.log', format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',
                            filemode='w', level=logging.DEBUG)
        logging.info("Total Test Samples '%s'" % (len(image_names)))
        logging.info("Correctly Classified Samples '%s'" % input_size)
        logging.info("Batch Size '%s'" % self.BATCH_SIZE)
        logging.info("Need to run for '%s' batches" % batch_evaluator)
        create_output(root_folder_to_save_img)
        for i in range(batch_evaluator):
            logging.info("Running For Batch '%s' " % (i + 1))
            # starting index of the batch
            index_from = self.BATCH_SIZE * i
            # ending index of the batch
            index_to = input_size if batch_evaluator - i == 1 else (i * self.BATCH_SIZE + self.BATCH_SIZE)
            samples_batch = original_images_list[index_from:index_to]
            image_names_batch = img_names_correctly_classified[index_from:index_to]
            labels_batch = y_[index_from:index_to]
            logging.info("Sample names in the batch - " + str(image_names_batch))
            logging.info("Corresponding Labels - " + str(labels_batch))
            # y_truth_batch and y_pred_batch will have data for only those samples where adversarial has been found
            adv_img_name_list_batch, y_truth_batch, y_pred_batch, y_adv_batch = \
                self.grid_search(samples_batch, image_names_batch, labels_batch, img_height, img_width, mapping_dict,
                                 root_folder_to_save_img, org_imag_pred_prob[index_from:index_to])
            y_truth.extend(y_truth_batch)
            if len(adv_img_name_list_batch) > 0:
                output_csv_generator_rot_and_trans(adv_img_name_list_batch, y_truth_batch, y_pred_batch, y_adv_batch,
                                                   root_folder_to_save_img)
        print("output.csv has been generated..")
        return rot_trans_class_overall_metrics_generator(root_folder_to_save_img, y_truth, y,
                                                         self.threshold, overall_testdata_pred_probs,
                                                         mapping_dict, self.experiments_filepath,
                                                         self.method_type)
