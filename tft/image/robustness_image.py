# authors: TFTAuthors@accenture.com
# Purpose: This module abstracts different methods to audit an image classifier model for Robustness

import tensorflow as tf
from . adversarial_patch import adv_patch
import json
import os
import csv
import math
import numpy as np
from . utils_image import get_imgs_clsfied_corr, read_img_names_labels_csv, read_images, save_images
from . utils_image import results_corr_clsfied_imgs, create_output, get_imgs_clsfied_corr_small_batch
from . utils_image import output_csv_generator, class_metrics_overall_csv_generator
import matplotlib.pyplot as plt


class AdversarialInputs:
    # Fast Gradient Sign Method, Carlini & Wagner Method to generate Adversarial examples
    # paper (FGSM): https://arxiv.org/pdf/1608.04644.pdf
    # paper (CW): https://arxiv.org/pdf/1412.6572.pdf)

    def __init__(self, model, path_to_org_imgs, img_label_csv_path, path_to_save_res, project_name,
                 class_index_json_path, num_classes, threshold=0.1, targeted=False, target_class=None):

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
        :param num_classes: an int - number of classes the model is trained on.
        :param threshold: a float value between 0 and 1. If the difference between the % of original examples correctly
               classified and % of adversarial examples correctly classified is greater than threshold, the metric of
               adversarial examples will be highlighted in red color in UI to indicate the model does not meet desired
               robustness. By default 0.1.
        :param targeted: Boolean - to specify targeted or non-targeted attack to generate Adversarial Examples
        :param target_class: an int - If targeted attack, specify integer value for target class to be used in cross
               entropy loss.
        """

        self.model = model
        self.threshold = threshold
        self.PATH_TO_ORG_IMAGES = path_to_org_imgs
        self.PATH_TO_SAVE_RESULTS = path_to_save_res
        self.PROJECT_NAME = project_name
        self.CLASS_INDEX_JSON_PATH = class_index_json_path
        self.IMAGE_LABELS_CSV_PATH = img_label_csv_path
        self.BATCH_SIZE = 32  # For processing test data in batches of 50 samples
        self.target_class = target_class
        self.targeted = targeted
        # number of classes the model is trained
        self.num_classes = num_classes
        # Code added to generate the settings.csv file
        with open(self.CLASS_INDEX_JSON_PATH) as index_class_mapping_file:
            self.mapping_dictionary = json.load(index_class_mapping_file)

    def fgsm(self, epsilon):

        """
                function that does the task of generating adversarial inputs using fast gradient sign method.
                :return a string message
        """
        method_type = "fastGradientSignMethod"

        root_folder_to_save_settings = os.path.join(self.PATH_TO_SAVE_RESULTS, self.PROJECT_NAME, 'Robustness',
                                                    'AdversarialInputs', method_type)

        if not os.path.isdir(root_folder_to_save_settings):
            os.makedirs(root_folder_to_save_settings, exist_ok=True)
        # settings.csv.
        with open(os.path.join(root_folder_to_save_settings, "settings.csv"), 'w', newline='') as output_file:
            output_writer = csv.writer(output_file, delimiter=',')

            output_writer.writerow(['epsilon', 'threshold', 'targeted', 'target_class'])
            output_writer.writerow([epsilon, self.threshold, self.targeted, "None"
                                    if self.target_class is None else self.target_class])
            output_writer.writerow(["Input variation parameter. [small value multiplied during the generation "
                                    "of perturbations]",
                                    "By default 0.1.  [a float value between 0 and 1. The metric of adversarial "
                                    "examples will be highlighted in red color to indicate the model does not meet "
                                    "desired robustness If the difference between the % of original examples correctly"
                                    " classified and % of adversarial examples correctly classified is greater than "
                                    "threshold.]",
                                    "Boolean True or False. By default False. [Is the attack targeted or untargeted? "
                                    "The default which is untargeted will try to make the label incorrect. Targeted "
                                    "will instead try to move in the direction of being more like y.]",
                                    "target_class: an int. [Specifies integer value for target class to be "
                                    "used in cross"
                                    " entropy loss if adversarial examples have to generated using targeted attack.]"])
            print("settings.csv has been generated..")
        experiments_filepath = os.path.join(self.PATH_TO_SAVE_RESULTS, self.PROJECT_NAME)

        # Assertion for targeted & non-targeted attacks

        if self.targeted:
            assert (self.target_class is not None), "Provide target_class (integer) for targeted attack"

        # image_names: List of image names like a.JPEG, b.JPEG etc..
        # y : List of true label names for images

        image_names, y = read_img_names_labels_csv(self.IMAGE_LABELS_CSV_PATH)

        root_folder_to_save_images = os.path.join(self.PATH_TO_SAVE_RESULTS, self.PROJECT_NAME,
                                                  'Robustness',
                                                  'AdversarialInputs', method_type, 'Results')
        # For Example, if PATH_TO_SAVE_RESULTS is "C:\", then root_folder_to_save_imgs will be
        #  "C:\Robustness\AdversarialInputs\Results"
        if not os.path.isdir(root_folder_to_save_images):
            os.makedirs(root_folder_to_save_images, exist_ok=True)

        # Method 2 - Get a batch of 50 samples, evaluate correctly classified images, generate adv examples and
        #  repeat until all test samples are processed. This method seem to be 5 min faster than method 1(see below)
        img_names_correctly_classified, y_, overall_testdata_pred_probs, org_imag_pred_prob = get_imgs_clsfied_corr(
            self.model,
            self.PATH_TO_ORG_IMAGES, image_names,
            self.model.image_size_height,
            self.model.image_size_width, y,
            self.CLASS_INDEX_JSON_PATH)
        samples_path_list = [os.path.join(self.PATH_TO_ORG_IMAGES, img_name) for img_name
                             in img_names_correctly_classified]

        input_size = len(samples_path_list)

        # batch_evaluator ==> Calculate number of iterations for batch_size; For example: if batch_size is 200 and if we
        # have 1000 images, then batch_evaluator is 5. And for 50K images batch_evaluator is 250

        batch_evaluator: int = math.ceil((input_size / self.BATCH_SIZE))
        create_output(root_folder_to_save_images)
        for i in range(batch_evaluator):
            # starting index of the batch
            index_from = self.BATCH_SIZE * i
            # ending index of the batch
            index_to = input_size if batch_evaluator - i == 1 else (i * self.BATCH_SIZE + self.BATCH_SIZE)

            image_names_batch = img_names_correctly_classified[index_from:index_to]
            y_batch = y_[index_from:index_to]
            unique_label_names = list(set(y_batch))
            samples_ndarray_batch = read_images(samples_path_list[index_from:index_to],
                                                self.model.image_size_height, self.model.image_size_width)
            if self.targeted:
                print("TARGET CLASS --- ", self.target_class)
                number_corrclsfied = len(image_names_batch)  # Number of correctly classified images
                indices = tf.zeros([number_corrclsfied], dtype=tf.int32) + self.target_class
                target = tf.one_hot(indices, self.num_classes, on_value=1.0, off_value=0.0)
                loss = tf.nn.softmax_cross_entropy_with_logits(labels=target, logits=self.model.logits)
            else:
                print("UNTARGETED")
                loss = self.model.logits

            grad, = tf.gradients(loss, self.model.x)  # Computing gradients
            mul = tf.scalar_mul(epsilon, tf.sign(grad))  # Multiplying the sign of gradients with epsilon
            adv_x = tf.add(self.model.x, mul)  # Adding the generated perturbation to original images
            pre_processed_batch = self.model.pre_process(samples_ndarray_batch)
            print(pre_processed_batch[0])
            adv_examples = self.model.sess.run(adv_x, feed_dict={self.model.x: pre_processed_batch})

            adv_images_reverse_preprocessed = self.model.rev_preprocess(adv_examples)
            save_images(unique_label_names, root_folder_to_save_images, image_names_batch, y_batch,
                        samples_ndarray_batch, adv_images_reverse_preprocessed)

            adv_img_pred_probs = self.model.predict(adv_images_reverse_preprocessed)
            # org_img_pred_probs_corrclsfied = self.model.predict(samples_ndarray_batch)
            org_img_pred_probs_corrclsfied = org_imag_pred_prob[index_from:index_to]
            output_csv_generator(org_img_pred_probs_corrclsfied, adv_img_pred_probs, self.mapping_dictionary,
                                 y_batch, image_names_batch, root_folder_to_save_images)
        return class_metrics_overall_csv_generator(overall_testdata_pred_probs, self.mapping_dictionary,
                                                   self.IMAGE_LABELS_CSV_PATH,
                                                   root_folder_to_save_images,
                                                   self.threshold, method_type, experiments_filepath)

    def cw(self, learning_rate=5e-3, num_iterations=200):
        """
                main function that does the task of generating adversarial inputs using carlini & wagner method.

                :return a string message
                """
        # Assertion for targeted & non-targeted attacks

        if self.targeted:
            assert (self.target_class is not None), "Provide target_class (integer) for targeted attack"
        method_type = "carliniWagnerMethod"
        root_folder_to_save_settings = os.path.join(self.PATH_TO_SAVE_RESULTS, self.PROJECT_NAME, 'Robustness',
                                                    'AdversarialInputs', method_type)

        if not os.path.isdir(root_folder_to_save_settings):
            os.makedirs(root_folder_to_save_settings, exist_ok=True)
        # settings.csv.
        with open(os.path.join(root_folder_to_save_settings, "settings.csv"), 'w', newline='') as output_file:
            output_writer = csv.writer(output_file, delimiter=',')

            output_writer.writerow(['threshold', 'targeted', 'target_class', 'learning_rate', 'num_iterations'])
            output_writer.writerow([self.threshold, self.targeted, "None" if self.target_class is None else
                                   self.target_class, learning_rate, num_iterations])
            output_writer.writerow(["By default 0.1.  [a float value between 0 and 1. The metric of adversarial "
                                    "examples will be highlighted in red color to indicate the model does not meet "
                                    "desired robustness if the difference between the % of original examples correctly"
                                    " classified and % of adversarial examples correctly classified is greater than "
                                    "threshold.]",
                                    "Boolean True or False. By default False. [Is the attack targeted or untargeted? "
                                    "The default which is untargeted will try to make the label incorrect. Targeted "
                                    "will instead try to move in the direction of being more like y.]",
                                    "target_class: an int. [Specifies integer value for target class to be used "
                                    "in cross"
                                    " entropy loss if adversarial examples have to generated using targeted attack.]",
                                    "By default 0.005. parameter for Adam Optimizer used to optimize the loss",
                                    "An integer value. number of times the loop is run to optimize the loss by "
                                    "updating tf.Variable"])
            print("settings.csv has been generated..")
        experiments_filepath = os.path.join(self.PATH_TO_SAVE_RESULTS, self.PROJECT_NAME)
        # image_names: List of image names like a.JPEG, b.JPEG etc..
        # y : List of true label names for images
        image_names, y = read_img_names_labels_csv(self.IMAGE_LABELS_CSV_PATH)
        root_folder_to_save_images = os.path.join(self.PATH_TO_SAVE_RESULTS, self.PROJECT_NAME, 'Robustness',
                                                  'AdversarialInputs', method_type, 'Results')
        # For Example, if PATH_TO_SAVE_RESULTS is "C:\", then root_folder_to_save_imgs will be
        #  "C:\Robustness\AdversarialInputs\Results"
        if not os.path.isdir(root_folder_to_save_images):
            os.makedirs(root_folder_to_save_images, exist_ok=True)

        # Method 2 - Get a batch of 50 samples, evaluate correctly classified images, generate adv examples and
        #  repeat until all test samples are processed. This method seem to be 5 min faster than method 1(see below)
        img_names_correctly_classified, y_, overall_testdata_pred_probs, org_imag_pred_prob = get_imgs_clsfied_corr(
            self.model,
            self.PATH_TO_ORG_IMAGES, image_names,
            self.model.image_size_height,
            self.model.image_size_width, y,
            self.CLASS_INDEX_JSON_PATH)
        samples_path_list = [os.path.join(self.PATH_TO_ORG_IMAGES, img_name) for img_name
                             in img_names_correctly_classified]
        input_size = len(samples_path_list)
        # batch_evaluator ==> Calculate number of iterations for batch_size; For example: if batch_size is 200 and if we
        # have 1000 images, then batch_evaluator is 5. And for 50K images batch_evaluator is 250
        batch_evaluator: int = math.ceil((input_size / self.BATCH_SIZE))
        create_output(root_folder_to_save_images)
        for i in range(batch_evaluator):
            # starting index of the batch
            index_from = self.BATCH_SIZE * i
            # ending index of the batch
            index_to = input_size if batch_evaluator - i == 1 else (i * self.BATCH_SIZE + self.BATCH_SIZE)
            image_names_batch = img_names_correctly_classified[index_from:index_to]
            y_batch = y_[index_from:index_to]
            unique_label_names = list(set(y_batch))
            samples_ndarray_batch = read_images(samples_path_list[index_from:index_to],
                                                self.model.image_size_height, self.model.image_size_width)
            number_corrclsfied = len(samples_ndarray_batch)  # number of correctly classified images
            samples_ndarray_corrclsfied_batch_01 = self.model.pre_process(samples_ndarray_batch)
            # Creating Variable of the shape as correctly classified batch of images
            perturb_image_variables = tf.Variable(initial_value=np.random.normal(
                size=(number_corrclsfied, self.model.image_size_height, self.model.image_size_width,
                      self.model.num_channels)), dtype=tf.float32)
            perturb_image = (tf.tanh(x=perturb_image_variables) + 1) / 2  # to ensure the image is between 0 & 1
            # Initializing the variable
            self.model.sess.run(tf.variables_initializer([perturb_image_variables]))

            if self.targeted:
                print("TARGET CLASS --> ", self.target_class)
                # One-hot matrix for targeted attack
                indices = tf.zeros([number_corrclsfied], dtype=tf.int32) + self.target_class  # target class
                targetclass_onehot = tf.one_hot(indices, self.num_classes, on_value=1.0, off_value=0.0)
                entropy_loss = tf.losses.softmax_cross_entropy(onehot_labels=targetclass_onehot,
                                                               logits=self.model.logits)

            else:
                print("UNTARGETED")
                entropy_loss = self.model.logits

            l2_loss = tf.nn.l2_loss(perturb_image - samples_ndarray_corrclsfied_batch_01)

            loss = entropy_loss + l2_loss

            # Optimising loss by updating the variable
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            train_step = optimizer.minimize(loss, var_list=[perturb_image_variables])
            # Initializing the variables of Adam Optimizer
            all_variables = tf.all_variables()
            is_not_initialized = self.model.sess.run([tf.is_variable_initialized(var) for var in all_variables])
            not_initialized_vars = [v for (v, f) in zip(all_variables, is_not_initialized) if not f]
            self.model.sess.run(tf.variables_initializer(not_initialized_vars))
            p_img = []
            for optStep in range(num_iterations):
                _, loss_value, loss_ce, loss_l2, p_img, p_img_logits = \
                    self.model.sess.run([train_step, loss, entropy_loss, l2_loss, perturb_image,
                                         self.model.logits],
                                        feed_dict={self.model.x: samples_ndarray_batch})

            # p_img *= 255.
            # p_img = p_img.astype(np.uint8)  # 0-255
            p_img = self.model.rev_preprocess(p_img)
            adv_img_pred_probs_corrclsfied = self.model.predict(p_img)

            save_images(unique_label_names, root_folder_to_save_images, image_names_batch, y_batch,
                        samples_ndarray_batch, p_img)
            org_img_pred_probs_corrclsfied = org_imag_pred_prob[index_from:index_to]
            output_csv_generator(org_img_pred_probs_corrclsfied, adv_img_pred_probs_corrclsfied,
                                 self.mapping_dictionary,
                                 y_batch, image_names_batch,
                                 root_folder_to_save_images)

        return class_metrics_overall_csv_generator(overall_testdata_pred_probs, self.mapping_dictionary,
                                                   self.IMAGE_LABELS_CSV_PATH,
                                                   root_folder_to_save_images,
                                                   self.threshold, method_type, experiments_filepath)


class AdversarialPatches:

    def __init__(self, model, path_to_org_imgs, img_label_csv_path, path_to_save_res, project_name,
                 class_index_json_path, num_labels, path_to_model, model_file_name, input_tensor_name,
                 logits_tensor_name, patch_block, range_model_input, patch_appearance='circle', batch_size=4,
                 learning_rate=5.0, scale_min=0.1, scale_max=1.0, max_rotation=22.5, test_scale=0.3, target_label=None,
                 threshold=0.1):
        """

        :param model: An object of a model class created by the user. Expected to have the parameters "self.sess",
               "self.x"(where x is a input placeholder) and "self.logits", "self.image_size_height",
               "self.image_size_width" and "self.num_channels"
        :param path_to_org_imgs: a string - a local absolute path to original images
        :param img_label_csv_path: a string - a local absolute path to a csv file that contains mapping between image
               names and corresponding labels. The column names of the csv file must be “ImageName” and “Label”
               respectively.
        :param path_to_save_res: a string - a local absolute path to save results of the audit
        :param class_index_json_path: a string - a local absolute path to a .json file that contains the index-class
                mapping [Example: https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json]
        :param num_labels: an integer to denote number of labels.
        :param path_to_model: a string - a local absolute path to model
        :param model_file_name: a string - name of model file of type(.h5 or .meta)
        :param input_tensor_name: a string - name of the input tensor of the model
        :param logits_tensor_name: a string - name of the logits tensor of the model
        :param patch_block: an integer - size of each patch_block, for example if 13 then each patch_block would be of
               size 13x13,and an image of 299 x299 would be divided into (299/13 x 299/13)= (23 x 23) such patch_blocks.
        :param range_model_input: a tuple of floats. Range of values such that reverse preprocess gives valid original
               images
        :param patch_appearance: a string ('circle' or 'rectangle'). By default 'circle'. Indicates how the patch
               appear on images.
        :param batch_size: an integer - the size of batch to train adversarial patches, by default 4.
        :param learning_rate: a float - learning rate for training the patch
        :param scale_min: a float - the minimum scale of patch w.r.t image while training, by default 0.1
        :param scale_max: a float - the maximum scale of patch w.r.t image while training, by default 1.0
        :param max_rotation: an float - the maximum rotation of patch(in degrees) w.r.t image while training and
               testing, by default 22.5 degrees
        :param test_scale: an float or a tuple of float - if integer then all the test patches will have scale
               =test_scale on the image,if tuple =(t1,t2), then all the test patches will have scales as random value
               between t1 and t2
        :param target_label: a string - the target label for patch, by default None.
        :param threshold: a float value between 0 and 1. If the difference between the fraction of original examples
               correctly classified and fraction of adversarial examples correctly classified is greater than threshold,
               the metric of adversarial examples will be highlighted in red color in UI to indicate the model does not
               meet desired robustness. By default 0.1.
        """

        self.model = model
        self.PATH_TO_ORG_IMAGES = path_to_org_imgs
        self.PATH_TO_SAVE_RESULTS = path_to_save_res
        self.PROJECT_NAME = project_name
        self.CLASS_INDEX_JSON_PATH = class_index_json_path
        self.IMAGE_LABELS_CSV_PATH = img_label_csv_path
        self.BATCH_SIZE = batch_size
        self.LEARNING_RATE = learning_rate
        self.NUM_LABELS = num_labels
        self.PATH_TO_MODEL = path_to_model
        self.MODEL_FILE_NAME = model_file_name
        self.INPUT_TENSOR_NAME = input_tensor_name
        self.LOGITS_TENSOR_NAME = logits_tensor_name
        self.PATCH_BLOCK = patch_block
        self.RANGE_MODEL_INPUT = range_model_input
        self.PATCH_APPEARANCE = patch_appearance
        self.SCALE_MIN = scale_min
        self.SCALE_MAX = scale_max
        self.MAX_ROTATION = max_rotation
        self.TEST_SCALE = test_scale
        self.TARGET_LABEL = target_label
        self.THRESHOLD = threshold

        # Code added to generate the settings.csv file
        self.root_folder_to_save_settings = os.path.join(self.PATH_TO_SAVE_RESULTS, self.PROJECT_NAME, 'Robustness',
                                                         'AdversarialPatches')

        if not os.path.isdir(self.root_folder_to_save_settings):
            os.makedirs(self.root_folder_to_save_settings, exist_ok=True)
        # settings.csv.
        if self.TARGET_LABEL is not None:
            with open(os.path.join(self.root_folder_to_save_settings, "settings.csv"), 'w', newline='') as output_file:
                output_writer = csv.writer(output_file, delimiter=',')
                output_writer.writerow(['Patch Block', 'Model Input Range', 'Patch Appearance', 'Batch Size',
                                       'Learning Rate', 'Target Label', 'Threshold'])
                output_writer.writerow([self.PATCH_BLOCK, str(self.RANGE_MODEL_INPUT).replace(",", " to"),
                                        self.PATCH_APPEARANCE,
                                        self.BATCH_SIZE, self.LEARNING_RATE, self.TARGET_LABEL, self.THRESHOLD])
                output_writer.writerow(['an integer - size of each patch block. For example if 13 then each patch block'
                                        ' would be of size 13x13 and an image of 299x299 would be divided into'
                                        ' (299/13 x 299/13)= (23 x 23) such patch blocks',
                                        'a tuple of floats. Range of values such that reverse preprocess gives valid '
                                        'original images',
                                        'a string ("circle" or "rectangle"). By default "circle". Indicates how the '
                                        'patch appear on images',
                                        'an integer - the size of batch to train adversarial patches. By default 4.',
                                        'a float - Learning rate used to train the patch',
                                        'a string - the target label for patch.',
                                        'By default 0.1.  [a float value between 0 and 1. The metric of adversarial'
                                        'examples will be highlighted in red color to indicate the model does not meet'
                                        'desired robustness if the difference between the % of original examples correctly'
                                        'classified and % of adversarial examples correctly classified is greater than'
                                        'threshold.]'])
            print("settings.csv has been generated..")

    def run(self):

        """
        main function that does the task of generating adversarial patches. Uses adversarial patches library in turn.

        :return  a string message whether the results were generated successfully or not
        """

        method_type = "adversarialPatchesValue"
        experiments_filepath = os.path.join(self.PATH_TO_SAVE_RESULTS, self.PROJECT_NAME)

        # image_names: List of image names like a.JPEG, b.JPEG etc..
        # y : List of true label names for images
        image_names, y = read_img_names_labels_csv(self.IMAGE_LABELS_CSV_PATH)

        root_folder_to_save_imgs = os.path.join(self.PATH_TO_SAVE_RESULTS, self.PROJECT_NAME, 'Robustness',
                                                'AdversarialPatches', 'Results')
        # For Example, if PATH_TO_SAVE_RESULTS is "C:\", then root_folder_to_save_imgs will be
        # "C:\<Project_Name>\Robustness\AdversarialPatches\Results"
        if not os.path.isdir(root_folder_to_save_imgs):
            os.makedirs(root_folder_to_save_imgs, exist_ok=True)

        samples_path_list = [os.path.join(self.PATH_TO_ORG_IMAGES, img_name) for img_name in image_names]

        samples_ndarray = read_images(samples_path_list,
                                      self.model.image_size_height, self.model.image_size_width)
        bs = 32
        batch_evaluator_predict: int = math.ceil((len(samples_path_list)/bs))
        overall_testdata_pred_probs = []
        
        for i in range(batch_evaluator_predict):
            index_from = bs * i
            # ending index of the batch
            index_to = len(samples_path_list) if batch_evaluator_predict - i == 1 else (i * bs + bs)

            samples_ndarray_batch = read_images(samples_path_list[index_from:index_to],
                                                self.model.image_size_height, self.model.image_size_width)
            org_pred_probs_batch = self.model.predict(samples_ndarray_batch)
            overall_testdata_pred_probs.extend(org_pred_probs_batch)

        img_names_corrclsfied, y_corrclsfied, samples_ndarray_corrclsfied, org_img_pred_probs_corrclsfied = \
            get_imgs_clsfied_corr_small_batch(samples_ndarray, overall_testdata_pred_probs, image_names, y,
                                              self.CLASS_INDEX_JSON_PATH)

        with open(self.CLASS_INDEX_JSON_PATH) as index_class_mapping_file:
            label_to_name = json.load(index_class_mapping_file)

        name_to_label = {v: k for k, v in label_to_name.items()}

        if self.TARGET_LABEL is None:
            all_labels = set(name_to_label.keys())
            our_labels = set(y_corrclsfied)
            target_label_choices = list(all_labels - our_labels)
            if len(target_label_choices) > 0:
                self.TARGET_LABEL = target_label_choices[int(np.random.uniform(0, len(target_label_choices)))]
            else:
                self.TARGET_LABEL = list(all_labels)[int(np.random.uniform(0, len(all_labels)))]
            # settings.csv
            with open(os.path.join(self.root_folder_to_save_settings, "settings.csv"), 'w', newline='') as output_file:
                output_writer = csv.writer(output_file, delimiter=',')
                output_writer.writerow(['Patch Block', 'Model Input Range', 'Patch Appearance', 'Batch Size',
                                       'Learning Rate', 'Target Label', 'Threshold'])
                output_writer.writerow([self.PATCH_BLOCK, str(self.RANGE_MODEL_INPUT).replace(",", " to"),
                                        self.PATCH_APPEARANCE,
                                        self.BATCH_SIZE, self.LEARNING_RATE, self.TARGET_LABEL, self.THRESHOLD])
                output_writer.writerow(['an integer - size of each patch block. For example if 13 then each patch block'
                                        ' would be of size 13x13 and an image of 299x299 would be divided into'
                                        ' (299/13 x 299/13)= (23 x 23) such patch blocks',
                                        'a tuple of floats. Range of values such that reverse preprocess gives valid '
                                        'original images',
                                        'a string ("circle" or "rectangle"). By default "circle". Indicates how the '
                                        'patch appear on images',
                                        'an integer - the size of batch to train adversarial patches. By default 4.',
                                        'a float - Learning rate to train the patch',
                                        'a string - the target label for patch.',
                                        'By default 0.1.  [a float value between 0 and 1. The metric of adversarial'
                                        'examples will be highlighted in red color to indicate the model does not meet'
                                        'desired robustness if the difference between the % of original examples correctly'
                                        'classified and % of adversarial examples correctly classified is greater than'
                                        'threshold.]'])

        print("\n\ntarget label is\n\n", self.TARGET_LABEL)

        samples_ndarray_corrclsfied_preprocessed = self.model.pre_process(samples_ndarray_corrclsfied)

        # adv_patches and transparent_patch are numpy array of sizes (no. of images)x(image size) and (image size)x4
        # respectively for example if number of images are 100 and size of each is (32x32x3) then
        # adv_patches is numpy array of size 100x32x32x3
        # and transparent_patch is numpy array of size 32x32x4, note that 4th channel is for transparent pixel values
        adv_patches, transparent_patch = adv_patch(self.model, self.TARGET_LABEL, name_to_label,
                                                   samples_ndarray_corrclsfied_preprocessed, self.NUM_LABELS,
                                                   self.PATH_TO_MODEL, self.MODEL_FILE_NAME, self.INPUT_TENSOR_NAME,
                                                   self.LOGITS_TENSOR_NAME, len(y_corrclsfied), self.PATCH_BLOCK,
                                                   self.RANGE_MODEL_INPUT, self.PATCH_APPEARANCE, self.BATCH_SIZE,
                                                   self.LEARNING_RATE, test_scale=self.TEST_SCALE)

        unique_label_names = list(set(y_corrclsfied))
        adv_patches_reverse_preprocessed = self.model.rev_preprocess(adv_patches)  # (adv_patches + 1) / 2

        save_images(unique_label_names, root_folder_to_save_imgs, img_names_corrclsfied, y_corrclsfied,
                    samples_ndarray_corrclsfied, adv_patches_reverse_preprocessed)
        # do something generic for +1/2

        adv_img_pred_probs_corrclsfied = []
        # print("SHAPE OF ADV_REV --- ", adv_patches_reverse_preprocessed.shape)
        batch_evaluator_predict: int = math.ceil((len(adv_patches_reverse_preprocessed) / bs))

        for i in range(batch_evaluator_predict):
            print("ITERATION IN PREDICT-2 --- ", i)
            print(len(adv_patches_reverse_preprocessed))
            index_from = bs * i
            # ending index of the batch
            index_to = len(adv_patches_reverse_preprocessed) if batch_evaluator_predict - i == 1 else (i * bs + bs)
            adv_ndarray_batch = adv_patches_reverse_preprocessed[index_from:index_to]
            
            adv_pred_probs_batch = self.model.predict(adv_ndarray_batch)
            adv_img_pred_probs_corrclsfied.extend(adv_pred_probs_batch)

        save_patch_path = os.path.join(root_folder_to_save_imgs, 'patch.png')
        plt.imsave(save_patch_path, transparent_patch)

        return results_corr_clsfied_imgs(overall_testdata_pred_probs, org_img_pred_probs_corrclsfied,
                                         adv_img_pred_probs_corrclsfied, self.CLASS_INDEX_JSON_PATH, y, y_corrclsfied,
                                         img_names_corrclsfied, root_folder_to_save_imgs, self.THRESHOLD, method_type,
                                         experiments_filepath)
