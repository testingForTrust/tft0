# author: TFTAuthors@accenture.com
# This file contains some utility methods that could come handy for integrating different methods to tft library.

import json
import matplotlib.image as mp_img
from keras.preprocessing import image
from PIL import Image
import csv
import os
import glob
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from collections import defaultdict
import pandas as pd
import math
import sys

# Class to throw a custom exception when the column names of test data csv is not as expected


class CsvColumnNameException(Exception):
    pass


def read_img_names_labels_csv(csv_path):
    """
    this function reads the image_name_label csv and returns a list of image names and list of corresponding labels

    :param csv_path: A string. Absolute path to csv
    :return: A list of image names and corresponding labels
    """
    df = pd.read_csv(csv_path)

    try:
        image_names_list = list(df['ImageName'])
        y = list(df['Label'])
    except KeyError:
        raise CsvColumnNameException(" The column names of image-name_label csv must be 'ImageName' and 'Label' ")

    return image_names_list, y


def read_images(path_list, img_height, img_width):
    """
    A function that reads raw samples from path_list [0 to 255] and stacks one below the other.
    Uses image module from Keras which takes care of all types of images. Internally uses Python Image Library (PIL).

    A raw JPEG image when read will have values in the range [0-255] & values of dtypes 'uint8'.Can be verified as below

    :param path_list: List of absolute paths to the image samples
    :param img_height: Height of the image input as required by the model
    :param img_width: Width of the image input as required by the model
    :return: A stacked array of numpy arrays of constant height and width. Shape (nSamples, h, w, c)
    """

    out = []
    for img_path in path_list:
        # Below line converts to a PIL Python Image Library format
        img = image.load_img(img_path, target_size=(img_height, img_width))
        x = image.img_to_array(img)  # Convert a PIL image to a numpy array
        x = np.expand_dims(x, axis=0)  # (1, dim1, dim2, channels) Type: float32
        out.append(x.astype('uint8'))
    return np.vstack(out)  # Stack the images one below the other (rows)


def resize_image(arr, img_height, img_width):
    """
    This function reshapes a given image array from its original shape to a shape as required by the model
    :param arr: A numpy nd array of 3 dimensions i.e. (h, w, c)
    :param img_height: Image height as required by the model under consideration
    :param img_width: Image width as required by the model under consideration
    :return: returns a reshaped numpy array of 3 dimensions
    """
    arr_pil = Image.fromarray((arr * 255).astype(np.uint8))  # Convert to a PIL Python Image Library format
    out = arr_pil.resize((img_height, img_width))
    a = image.img_to_array(out)
    return a


def read_orig_dim_images_from_list_of_numpy_arrays(list_of_images, target_img_height, target_img_width):
    """
    This function reads original images of different dimensions from a list containing images in numpy array and
    reshapes to a height and width (as required by the NN model)

    :param list_of_images: List of absolute paths to the image samples
    :param target_img_height: Height of the image input as required by the model
    :param target_img_width: Width of the image input as required by the model
    :return: A stacked array of numpy arrays of constant height and width (nSamples, h, w, c)
    """
    out = []
    for arr in list_of_images:
        x = resize_image(arr / np.max(arr), target_img_height, target_img_width)
        x = np.expand_dims(x, axis=0)
        out.append(x)
    return np.vstack(out)


def save_images(unique_class_names, root_folder_to_save_images, img_names, y, original_images, perturbed_images):
    """
     A function that create folders by class names and save corresponding original and perturbed images
    :param unique_class_names: A list of unique classes trained on the model
    :param root_folder_to_save_images: A string. Absolute path to save respective original and perturbed images
    :param img_names: A list of image sample names
    :param y: A list of labels for corresponding image sample names
    :param original_images: A list of numpy arrays representing image samples
    :param perturbed_images: A list of numpy arrays representing perturbed image samples
    :return: None
    """
    original_images = original_images / np.max(original_images)
    perturbed_images = perturbed_images / np.max(perturbed_images)

    if not os.path.isdir(root_folder_to_save_images):
        os.makedirs(root_folder_to_save_images, exist_ok=True)
    for class_names in unique_class_names:
        perturbed_images_save_path = os.path.join(root_folder_to_save_images, class_names, 'perturbed')
        original_images_save_path = os.path.join(root_folder_to_save_images, class_names, 'original')
        if not os.path.isdir(perturbed_images_save_path):
            os.makedirs(perturbed_images_save_path, exist_ok=True)
        if not os.path.isdir(original_images_save_path):
            os.makedirs(original_images_save_path, exist_ok=True)

    for name_of_image, label, original_image, adversarial_image in zip(img_names, y, original_images, perturbed_images):
        absolute_path_perturbed_image = os.path.join(root_folder_to_save_images, label, 'perturbed', name_of_image)
        absolute_path_orig_image = os.path.join(root_folder_to_save_images, label, 'original', name_of_image)
        perturbed_image = adversarial_image.copy()
        mp_img.imsave(absolute_path_orig_image, original_image)
        mp_img.imsave(absolute_path_perturbed_image, perturbed_image)


def result_message(file_path):
    """
    A function that returns a success message or a failure message
    :param file_path: Path to result csv files
    :return: A string message
    """
    if (os.path.isfile(os.path.join(file_path, "overallmetrics.csv"))) and \
            (os.path.isfile(os.path.join(file_path, "output.csv"))) and \
            (os.path.isfile(os.path.join(file_path, "classmetrics.csv"))):
        return 'Results have been generated successfully'
    else:

        # This is will delete settings.csv file if the method fails to generate the results successfully

        settings_filepath = file_path[:(file_path.find('Results'))]
        csv_files = [file for file in glob.glob(os.path.join(settings_filepath, '*.csv'))]
        for file in csv_files:
            os.remove(file)
        print("Settings files deleted")
        return 'Some error while generating Adversarial results'


def create_output(root_folder_to_save_csv):
    """
    A function that creates an output csv with only columns in it.

    :param root_folder_to_save_csv: Path to save the csv.
    :return: None
    """
    df = pd.DataFrame(columns=['IMAGE Name', 'Original Class Name', 'Predictions On Original Images',
                               'Predictions On Perturbed Images'])
    df.to_csv(os.path.join(root_folder_to_save_csv, 'output.csv'), index=False)


def output_csv_generator(org_pred_probs_clscorr, perturbed_pred_probs,
                         mapping_dict, y_, img_names, root_folder_to_save_images):
    """
    A function that appends the image information (correctly classified and adversarial) to output csv
    for one batch.
    :param org_pred_probs_clscorr: List of predictions of correctly classified images for one batch.
    :param perturbed_pred_probs: List of prediction on perturbed images for one batch.
    :param mapping_dict: A dictionary - mapped to class label with index.
    :param y_: Class labels for one batch.
    :param img_names: Image names for one batch.
    :param root_folder_to_save_images: Path contains the output.csv.
    :return: None
    """
    original_image_predicted_classes = []
    perturbed_image_predicted_classes = []
    for org_preds, perturbed_preds in zip(org_pred_probs_clscorr, perturbed_pred_probs):
        org_top_predicted_label_index = org_preds.argsort()[-1]
        perturbed_top_predicted_label_index = perturbed_preds.argsort()[-1]
        original_image_predicted_classes.append(mapping_dict[str(org_top_predicted_label_index)])
        perturbed_image_predicted_classes.append(mapping_dict[str(perturbed_top_predicted_label_index)])

    with open(os.path.join(root_folder_to_save_images, "output.csv"), 'a+', newline='') as output_file:
        output_writer = csv.writer(output_file, delimiter=',')
        for i in zip(img_names, y_, original_image_predicted_classes, perturbed_image_predicted_classes):
            output_writer.writerow(i)


def class_metrics_overall_csv_generator(org_pred_probs_full_data, mapping_dict, image_label_csv_path,
                                        root_folder_to_save_images, threshold,
                                        method_type, experiments_filepath):
    """
    A function to generate classmetrics.csv and overallmetrics.csv for a particular method.

    :param org_pred_probs_full_data: List of predictions for all the images.
    :param mapping_dict: A dictionary - mapped to class label with index.
    :param image_label_csv_path: a string - a local absolute path to a csv file that contains mapping
    between image names and corresponding labels. The column names of the csv file must be “ImageName”
    and “Label” respectively.
    :param root_folder_to_save_images: Path to output.csv.
    :param threshold: A float value between 0 and 1. If the difference between the % of original
    examples correctly classified and % of adversarial examples correctly classified is greater
    than threshold, the metric of adversarial examples will be highlighted in red color in UI to
    indicate the model does not meet desired robustness. By default 0.1.
    :param method_type: A string referring to method name.
    :param experiments_filepath: Path to save experiments.csv file.
    :return: A string message indicating success or failure.
    """
    original_image_predicted_classes_full_data = []
    df = pd.read_csv(os.path.join(root_folder_to_save_images, "output.csv"))
    perturbed_image_predicted_classes = list(df['Predictions On Perturbed Images'])
    for org_preds_full_data in org_pred_probs_full_data:
        top_predicted_label_index = org_preds_full_data.argsort()[-1]
        original_image_predicted_classes_full_data.append(mapping_dict[str(top_predicted_label_index)])

    y_ = list(df['Original Class Name'])
    y = list(pd.read_csv(image_label_csv_path)['Label'])
    # for i,j in zip(orgLabels, predLabels):
    #     if(i==j):
    #         Y_.append(j)

    num_adv_samples_per_class = defaultdict(int)
    for label in y_:
        num_adv_samples_per_class[label] += 1

    unique_labels = list(num_adv_samples_per_class.keys())
    metrics_org_examples = precision_recall_fscore_support(
        y, original_image_predicted_classes_full_data, labels=unique_labels)

    support_org_examples = metrics_org_examples[3]
    conf_matrix_perturbed_imgs = confusion_matrix(y_, perturbed_image_predicted_classes, labels=unique_labels)
    conf_matrix_overall_imgs = confusion_matrix(y, original_image_predicted_classes_full_data, labels=unique_labels)
    perturbed_ex_correctly_classified_per_class = []
    test_data_correctly_classified_per_class = []
    perturbed_ex_correctly_classified_per_class.extend(list(conf_matrix_perturbed_imgs.diagonal()))
    test_data_correctly_classified_per_class.extend(list(conf_matrix_overall_imgs.diagonal()))

    # perturbed examples correctly classified divided by original number of examples in each class
    percent_of_adv_exmpls_classified_correctly = [j / i for i, j in zip(support_org_examples,
                                                                        perturbed_ex_correctly_classified_per_class)]
    # test data examples correctly classified divided by original number of samples in each class
    percent_of_test_data_classified_correctly = [n / m for m, n in zip(support_org_examples,
                                                                       test_data_correctly_classified_per_class)]
    diff_bool = []
    for m, n in zip(percent_of_test_data_classified_correctly, percent_of_adv_exmpls_classified_correctly):
        if n == 0:
            diff_bool.append('TRUE')
        else:
            diff_bool.append((m - n) > threshold)
    with open(os.path.join(root_folder_to_save_images, 'classmetrics.csv'), 'w', newline='') as classmetrics_file:
        classmetrics_writer = csv.writer(classmetrics_file, delimiter=',')
        classmetrics_writer.writerow(
            ['ClassNames', 'Number Of Examples In Each Class(Overall Test Data)',
             'Number Of Test Data Correctly Classified(Overall)', 'Perturbed Examples Correctly Classified',
             'Accuracy-TestData', 'Accuracy-PerturbedExamples', 'Diff-Accuracy'])
        for row in zip(unique_labels, support_org_examples, test_data_correctly_classified_per_class,
                       perturbed_ex_correctly_classified_per_class, percent_of_test_data_classified_correctly,
                       percent_of_adv_exmpls_classified_correctly, diff_bool):
            classmetrics_writer.writerow(row)

    print("classwisemetrics.csv has been generated..")
    # Overall Metrics
    # calculate overall true positives
    conf_matrix_full_testdata = confusion_matrix(y, original_image_predicted_classes_full_data, labels=list(set(y)))
    true_positives_overall_testdata = 0
    for d in range(conf_matrix_full_testdata.shape[0]):
        true_positives_overall_testdata += conf_matrix_full_testdata[d][d]

    accuracy_overall_testdata = true_positives_overall_testdata / len(y)
    accuracy_perturbed_examples = sum(perturbed_ex_correctly_classified_per_class) / len(y)
    with open(os.path.join(root_folder_to_save_images, 'overallmetrics.csv'), 'w', newline='') as overall_metrics_file:
        overall_metrics_writer = csv.writer(overall_metrics_file, delimiter=',')
        overall_metrics_writer.writerow(['Metrics', 'OnOriginalExamples', 'OnPerturbedExamples', 'Diff-Accuracy'])
        overall_metrics_writer.writerow(['Accuracy', accuracy_overall_testdata, accuracy_perturbed_examples,
                                         accuracy_overall_testdata - accuracy_perturbed_examples > threshold])

    print("overallmetrics.csv has been generated..")
    generate_experiments(experiments_filepath, method_type)
    return result_message(root_folder_to_save_images)


def results_corr_clsfied_imgs(org_pred_probs_full_data, org_pred_probs_clscorr, perturbed_pred_probs,
                              class_index_json_path, y, y_, img_names, root_folder_to_save_images, threshold,
                              method_type, experiments_file_path):
    """
     A function that generates audit result files based on predicted probabilities on correctly classified original
     images.

    :param org_pred_probs_full_data: Array of arrays of predictions for each original image sample.
           i.e.  [[p(class0/image0),p(class1/image0),....,p(classN/image0)], [p(class0/image1),
           p(class1/image1),....,p(classN/image1)],......., [p(class0/imageN),p(class1/imageN),....,p(classN/imageN)]]
    :param org_pred_probs_clscorr: Array of arrays of predictions for each image sample in test set
          (where test sample is correctly classified by the model).
    :param perturbed_pred_probs: Array of arrays of predictions for each perturbed image sample.
    :param class_index_json_path: A string - a local absolute path to a .json file that contains the index-class mapping
          [Example: https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json]
    :param y: A list of labels for corresponding image sample names
    :param y_: A list of labels for corresponding image samples correctly classified by the model.
    :param img_names: A list of image names correctly classified by the model.
    :param root_folder_to_save_images: A string. Absolute path to save result csv files
    :param threshold: a float value between 0 and 1. If the difference between the % of original examples correctly
           classified and % of adversarial examples correctly classified is greater than threshold, the metric of
           adversarial examples will be highlighted in red color in UI to indicate the model does not meet desired
           robustness. By default 0.1.
    :param method_type : A string value that generates the experiments based on the method run by the user.
    :param experiments_file_path: Path to generate the experiments.csv file
    :return: A string message indicating success or failure.
    """

    original_image_predicted_classes_full_data = []

    original_image_predicted_classes = []

    perturbed_image_predicted_classes = []

    with open(class_index_json_path) as index_class_mapping_file:
        mapping_dictionary = json.load(index_class_mapping_file)

    for org_preds, perturbed_preds in zip(org_pred_probs_clscorr, perturbed_pred_probs):
        org_top_predicted_label_index = org_preds.argsort()[-1]

        perturbed_top_predicted_label_index = perturbed_preds.argsort()[-1]
        original_image_predicted_classes.append(mapping_dictionary[str(org_top_predicted_label_index)])
        perturbed_image_predicted_classes.append(mapping_dictionary[str(perturbed_top_predicted_label_index)])

    for org_preds_full_data in org_pred_probs_full_data:
        top_predicted_label_index = org_preds_full_data.argsort()[-1]
        original_image_predicted_classes_full_data.append(mapping_dictionary[str(top_predicted_label_index)])

    with open(os.path.join(root_folder_to_save_images, "output.csv"), 'w', newline='') as output_file:
        output_writer = csv.writer(output_file, delimiter=',')
        output_writer.writerow(
            ['IMAGE Name', 'Original Class Name', 'Predictions On Original Images', 'Predictions On Perturbed Images'])
        for i in zip(img_names, y_, original_image_predicted_classes, perturbed_image_predicted_classes):
            output_writer.writerow(i)
    print("output.csv has been generated..")

    # Classwise metrics

    num_samples_per_class = defaultdict(int)
    for a_label in y:
        num_samples_per_class[a_label] += 1  # Example : { 'classA': 100 , 'classB': 99,...., 'classN': 89 }

    num_adv_samples_per_class = defaultdict(int)
    for label in y_:
        num_adv_samples_per_class[label] += 1

    # num of adversarial examples found for each class
    # num_of_adv_examples_found = list(num_adv_samples_per_class.values())

    unique_labels = list(num_adv_samples_per_class.keys())

    # Y_ ==> only labels for classes where correctly classified.Y==>Overall Data.
    # We need report only for correctly classified. Hence 'labels' argument has been assigned with classes where
    # samples have been correctly classified by the model.
    metrics_org_examples = precision_recall_fscore_support(
        y, original_image_predicted_classes_full_data, labels=unique_labels)
    # recall_org_examples = metrics_org_examples[1]
    support_org_examples = metrics_org_examples[3]

    # To get true positives on perturbed examples
    conf_matrix_perturbed_imgs = confusion_matrix(y_, perturbed_image_predicted_classes, labels=unique_labels)
    conf_matrix_overall_imgs = confusion_matrix(y, original_image_predicted_classes_full_data, labels=unique_labels)

    # Get diagonal elements from the matrix to get true positives
    perturbed_ex_correctly_classified_per_class = []
    for i in range(conf_matrix_perturbed_imgs.shape[0]):
        perturbed_ex_correctly_classified_per_class.append(conf_matrix_perturbed_imgs[i][i])

    test_data_correctly_classified_per_class = []
    for j in range(conf_matrix_overall_imgs.shape[0]):
        test_data_correctly_classified_per_class.append(conf_matrix_overall_imgs[j][j])

    # perturbed examples correctly classified divided by original number of examples in each class
    percent_of_adv_exmpls_classified_correctly = [j / i for i, j in zip(support_org_examples,
                                                                        perturbed_ex_correctly_classified_per_class)]
    # test data examples correctly classified divided by original number of samples in each class
    percent_of_test_data_classified_correctly = [n / m for m, n in zip(support_org_examples,
                                                                       test_data_correctly_classified_per_class)]

    diff_bool = []
    for m, n in zip(percent_of_test_data_classified_correctly, percent_of_adv_exmpls_classified_correctly):
        if n == 0:
            diff_bool.append('TRUE')
        else:
            diff_bool.append((m - n) > threshold)
    with open(os.path.join(root_folder_to_save_images, 'classmetrics.csv'), 'w', newline='') as classmetrics_file:
        classmetrics_writer = csv.writer(classmetrics_file, delimiter=',')
        classmetrics_writer.writerow(
            ['ClassNames', 'Number Of Examples In Each Class(Overall Test Data)',
             'Number Of Test Data Correctly Classified(Overall)', 'Perturbed Examples Correctly Classified',
             'Accuracy-TestData', 'Accuracy-PerturbedExamples', 'Diff-Accuracy'])
        for row in zip(unique_labels, support_org_examples, test_data_correctly_classified_per_class,
                       perturbed_ex_correctly_classified_per_class, percent_of_test_data_classified_correctly,
                       percent_of_adv_exmpls_classified_correctly, diff_bool):
            classmetrics_writer.writerow(row)

    print("classwisemetrics.csv has been generated..")

    # Overall Metrics
    # calculate overall true positives
    conf_matrix_full_testdata = confusion_matrix(y, original_image_predicted_classes_full_data, labels=list(set(y)))
    true_positives_overall_testdata = 0
    for d in range(conf_matrix_full_testdata.shape[0]):
        true_positives_overall_testdata += conf_matrix_full_testdata[d][d]

    accuracy_overall_testdata = true_positives_overall_testdata / len(y)
    accuracy_perturbed_examples = sum(perturbed_ex_correctly_classified_per_class) / len(y)
    with open(os.path.join(root_folder_to_save_images, 'overallmetrics.csv'), 'w', newline='') as overall_metrics_file:
        overall_metrics_writer = csv.writer(overall_metrics_file, delimiter=',')
        overall_metrics_writer.writerow(['Metrics', 'OnOriginalExamples', 'OnPerturbedExamples', 'Diff-Accuracy'])
        overall_metrics_writer.writerow(['Accuracy', accuracy_overall_testdata, accuracy_perturbed_examples,
                                         accuracy_overall_testdata - accuracy_perturbed_examples > threshold])

    print("overallmetrics.csv has been generated..")

    # Generate experiments.csv file.
    generate_experiments(experiments_file_path, method_type)

    # Check if the result files have been generated and the return a message

    return result_message(root_folder_to_save_images)


def output_csv_generator_rot_and_trans(img_names, y_truth_adv_found, y_pred_adv_found, y_pred_adv,
                                       root_folder_to_save_images):
    """
    A function that appends the image information (correctly classified and adversarial) to output csv
    for one batch.

    :param img_names: Image names for one batch.
    :param y_truth_adv_found: List of original class labels for one batch where adversarial is found.
    :param y_pred_adv_found: List of predicted class labels on original images for one batch.
    :param y_pred_adv: List of predicted class labels on perturbed images for one batch.
    :param root_folder_to_save_images: Path to save the csv files.
    :return: None
    """
    if not os.path.isdir(root_folder_to_save_images):
        os.makedirs(root_folder_to_save_images, exist_ok=True)
    # output.csv
    with open(os.path.join(root_folder_to_save_images, "output.csv"), 'a+', newline='') as output_file:
        output_writer = csv.writer(output_file, delimiter=',')
        for i in zip(img_names, y_truth_adv_found, y_pred_adv_found, y_pred_adv):
            output_writer.writerow(i)


def rot_trans_class_overall_metrics_generator(root_folder_to_save_images, y_truth_adv_found, y, threshold,
                                              overall_testdata_pred_probs, mapping_dict, experiments_filepath,
                                              method_type):
    """
    A function to generate classmetrics.csv and overallmatrics.csv for rRotationTranslation method.

    :param root_folder_to_save_images: Path to save the csv files.
    :param y_truth_adv_found: List of original class labels where adversarial is found.
    :param y: A list of labels corresponding to image_names.
    :param threshold: A float value between 0 and 1. If the difference between the % of original
    examples correctly classified and % of adversarial examples correctly classified is greater
    than threshold, the metric of adversarial examples will be highlighted in red color in UI to
    indicate the model does not meet desired robustness. By default 0.1.
    :param overall_testdata_pred_probs: A List of predicted probabilities for all image samples.
    :param mapping_dict: A dictionary - mapped to class label with index.
    :param experiments_filepath: Path where the experiments.csv file is generated.
    :param method_type: Type of the experiment run.
    :return: A string message indicating success or failure.
    """

    num_samples_per_class = defaultdict(int)
    for a_label in y:
        num_samples_per_class[a_label] += 1  # Example : { 'classA': 100 , 'classB': 99,...., 'classN': 89 }

    num_adv_samples_per_class = defaultdict(int)
    for label in y_truth_adv_found:
        num_adv_samples_per_class[label] += 1  # Example : { 'classA': 100 , 'classB': 99,...., 'classN': 89 }

    num_of_adv_examples_found = list(num_adv_samples_per_class.values())
    classes = list(num_adv_samples_per_class.keys())

    classes_all = list(num_samples_per_class.keys())

    num_samples_per_class_adv_found = []
    for c in classes:
        num_samples_per_class_adv_found.append(num_samples_per_class[c])

    testdata_pred_classes = []
    for org_preds in overall_testdata_pred_probs:
        org_top_predicted_label_index = org_preds.argsort()[-1]
        testdata_pred_classes.append(mapping_dict[str(org_top_predicted_label_index)])

    try:
        conf_mat_testdata = confusion_matrix(y, testdata_pred_classes, labels=classes)
    except ValueError :
        print("No adversarial examples were found in the current attack space. Try with different rotation & "
              "translation configurations.")
        sys.exit()
        
    testdata_corr_clsfied_per_cls = []
    for j in range(conf_mat_testdata.shape[0]):
        testdata_corr_clsfied_per_cls.append(conf_mat_testdata[j][j])

    per_of_adv_exmpls_corr_clsfied = [(c - b) / a for a, b, c in zip(num_samples_per_class_adv_found,
                                                                     num_of_adv_examples_found,
                                                                     testdata_corr_clsfied_per_cls)]

    # test data examples correctly classified divided by original number of samples in each class
    per_of_testdata_corr_clsfied = [m / n
                                    for m, n in zip(testdata_corr_clsfied_per_cls, num_samples_per_class_adv_found)]

    diff_bool = []
    for i, j in zip(per_of_testdata_corr_clsfied, per_of_adv_exmpls_corr_clsfied):
        if j == 0:
            diff_bool.append('TRUE')
        else:
            diff_bool.append((i - j) > threshold)

    with open(os.path.join(root_folder_to_save_images, 'classmetrics.csv'), 'w', newline='') as classmetrics_file:
        classmetrics_writer = csv.writer(classmetrics_file, delimiter=',')
        classmetrics_writer.writerow(
            ['ClassNames', 'Number Of Examples In Each Class(Overall Test Data)',
             'How Many Adversarial Examples Found?', 'Accuracy-OriginalExample', 'Accuracy-PerturbedExample',
             'Diff-Accuracy', 'Number Of Test Data Correctly Classified(Overall)'])
        for row in zip(classes, num_samples_per_class_adv_found, num_of_adv_examples_found,
                       per_of_testdata_corr_clsfied, per_of_adv_exmpls_corr_clsfied, diff_bool,
                       testdata_corr_clsfied_per_cls):
            classmetrics_writer.writerow(row)
    print("classmetrics.csv has been generated..")

    conf_mat_testdata_1 = confusion_matrix(y, testdata_pred_classes, labels=classes_all)
    testdata_corr_clsfied_= []
    for j in range(conf_mat_testdata_1.shape[0]):
        testdata_corr_clsfied_.append(conf_mat_testdata_1[j][j])

    acc_overall_testdata = sum(testdata_corr_clsfied_) / len(y)
    acc_rot_trans = (sum(testdata_corr_clsfied_per_cls) -
                     sum(num_of_adv_examples_found))/sum(num_samples_per_class_adv_found)

    with open(os.path.join(root_folder_to_save_images, "overallmetrics.csv"), 'w', newline='') as overall_metrics_file:
        overall_metrics_writer = csv.writer(overall_metrics_file, delimiter=',')
        overall_metrics_writer.writerow(['Metrics', 'OnOriginalExamples', 'OnAdversarialExamples', 'Diff'])
        overall_metrics_writer.writerow(['Accuracy', acc_overall_testdata, acc_rot_trans,
                                         (acc_overall_testdata - acc_rot_trans) > threshold])
    print("overallmetrics.csv has been generated..")
    generate_experiments(experiments_filepath, method_type)
    return result_message(root_folder_to_save_images)


def get_imgs_clsfied_corr_small_batch(samples_ndarray_batch, pred_probs_batch, img_names_batch, y_batch,
                                      class_index_json_path):
    """
    This function takes a batch of image arrays and returns data pertaining to those correctly classified by the model

    :param samples_ndarray_batch: A batch of ndarrays representing images
    :param pred_probs_batch: Array of arrays of predictions for each image sample in the batch
    :param img_names_batch: A list of image names like a.jpeg, b.jpeg etc..
    :param y_batch: A list labels corresponding to images in batch
    :param class_index_json_path: A string - a local absolute path to a .json file that contains the index-class mapping
          [Example: https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json]
    :return: returns 4 items.
                1. A list of image names correctly classified by the model
                2. A list of labels correctly classified by the model
                3. A numpy array of shape (nSamples, h, w, c) corresponding to correctly classified image samples
                4. An array of predicted probabilities corresponding to correctly classified image samples
    """

    with open(class_index_json_path) as index_class_mapping_file:
        mapping_dictionary = json.load(index_class_mapping_file)

    y_ = []
    samples_corrclsfied = []
    img_names_corrclsfied = []
    pred_probs_corrclsfied = []

    for sample, preds, img, label in zip(samples_ndarray_batch, pred_probs_batch, img_names_batch, y_batch):
        top_predicted_label_index = preds.argsort()[-1]
        predicted_class = mapping_dictionary[str(top_predicted_label_index)]
        if predicted_class.lower() == label.lower():
            samples_corrclsfied.append(np.expand_dims(sample, axis=0))  # shape (1, h, w, c)
            y_.append(label)
            img_names_corrclsfied.append(img)
            pred_probs_corrclsfied.append(preds)
    return img_names_corrclsfied, y_, np.vstack(samples_corrclsfied), pred_probs_corrclsfied


def get_imgs_clsfied_corr(model, path_to_org_images, image_names, image_height, image_width, y, class_index_json_path):
    """
    This function takes test data and returns only those classified correctly

    :param model: An object of a model class created by the user. Expected to have the parameters "self.sess", "self.x"
           and "self.logits", "self.image_size_height", "self.image_size_width" and "self.num_channels"
    :param path_to_org_images: A string - a local absolute path to original images
    :param image_names: A list of image names like a.jpeg, b.jpeg etc..
    :param image_height: Image height as required by the model
    :param image_width: Image width as required by the model
    :param y: A list of labels corresponding to image_names
    :param class_index_json_path: A string - a local absolute path to a .json file that contains the index-class mapping
          [Example: https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json]
    :return: 4 items.
            1. A list of image names correctly classified by the model
            2. A list of labels correctly classified by the model
            3. An array of predicted probabilities for all image samples.
            4. An array of predicted probabilities corresponding to correctly classified image samples.
    """

    samples_path_list = [os.path.join(path_to_org_images, img_name) for img_name in image_names]
    # samples_ndarray_classified_correctly = []
    y_ = []  # labels corresponding to images classified correctly
    img_names_corrclsfied = []
    org_pred_probs_corrclsfied = []
    org_pred_probs = []  # on overall data

    batch_size = 32
    input_size = len(samples_path_list)

    # batch_evaluator ==> Calculate number of iterations for batch_size; For example: if batch_size is 200 and if we
    # have 1000 images, then batch_evaluator is 5. And for 50K images batch_evaluator is 250
    batch_evaluator = int(math.ceil((input_size / batch_size)))

    with open(class_index_json_path) as index_class_mapping_file:
        mapping_dictionary = json.load(index_class_mapping_file)

    for i in range(batch_evaluator):
        image_names_batch = image_names[batch_size * i:i * batch_size + batch_size]
        y_batch = y[batch_size * i:i * batch_size + batch_size]
        pred_probs_corrclsfied_batch = []
        samples_ndarray = read_images(samples_path_list[batch_size * i:i * batch_size + batch_size], image_height,
                                      image_width)

        org_pred_probs_batch = model.predict(samples_ndarray)

        for sample, preds, img, label in zip(samples_ndarray, org_pred_probs_batch, image_names_batch, y_batch):
            top_predicted_label_index = preds.argsort()[-1]
            predicted_class = mapping_dictionary[str(top_predicted_label_index)]
            if predicted_class.lower() == label.lower():
                # samples_ndarray_classified_correctly.append(np.expand_dims(sample, axis=0))
                y_.append(label)
                img_names_corrclsfied.append(img)
                pred_probs_corrclsfied_batch.append(preds)
        org_pred_probs_corrclsfied.extend(pred_probs_corrclsfied_batch)
        org_pred_probs.extend(org_pred_probs_batch)

    # percent_of_original_examples_classified_correctly = len(y)/len(y_)

    # return np.vstack(samples_ndarray_classified_correctly), img_names_corrclsfied, y_, org_pred_probs

    return img_names_corrclsfied, y_, org_pred_probs, org_pred_probs_corrclsfied


def generate_experiments(experiments_filepath, method_type):
    """
    This function generates experiments.csv file based on the methods run by the user

    :param experiments_filepath: Path where the experiments.csv file is generated.
    :param method_type: Type of the experiment run.
    :return: None
    """

    if os.path.isfile(os.path.join(experiments_filepath, "experiments.csv")):
        experiment_data(experiments_filepath, method_type)
    else:
        with open(os.path.join(experiments_filepath, "experiments.csv"), 'w', newline='') as experiments_file:
            output_writer = csv.writer(experiments_file, delimiter=',')
            output_writer.writerow(
                ['generalizaionValue', 'rotationtranslationValue', 'fourierfilteringValue', 'colortograyscaleValue',
                 'contrastValue', 'additivenoiseValue',
                 'eidolonnoiseValue', 'modelcomplexityValue', 'robustnessValue', 'fastGradientSignMethod',
                 'carliniWagnerMethod',
                 'adversarialPatchesValue', 'discriminationValue', 'explainabilityValue'])
            output_writer.writerow(
                ['False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False',
                 'False', 'False', 'False'])
            experiments_file.close()
            experiment_data(experiments_filepath, method_type)
    print("experiments.csv has been generated..")


def experiment_data(experiments_filepath, method_type):
    """
     This method adds values to the csv based on the type of the ecperiment run by the user.
    :param experiments_filepath: Path where the experiments.csv file is generated.
    :param method_type: Type of the experiment run.
    :return: None
    """

    df = pd.read_csv(os.path.join(experiments_filepath, "experiments.csv"))
    df.at[0, method_type] = 'True'
    if method_type == 'rotationtranslationValue' or \
            method_type == 'fourierfilteringValue' or \
            method_type == 'colortograyscaleValue' or \
            method_type == 'contrastValue' or \
            method_type == 'additivenoiseValue' or \
            method_type == 'eidolonnoiseValue':
        df.at[0, 'generalizaionValue'] = 'True'
    if method_type == 'adversarialExamplesValue' or method_type == 'adversarialPatchesValue' \
            or method_type == 'fastGradientSignMethod' or method_type == 'carliniWagnerMethod':
        df.at[0, 'robustnessValue'] = 'True'
    df.to_csv(os.path.join(experiments_filepath, "experiments.csv"), index=False)


def result_message_performance_metrics(file_path):
    """
    A function that returns a success message or a failure message (for performance metrics of model)
    :param file_path: path to result csv files
    :return: a string message
    """
    if(os.path.isfile(os.path.join(file_path, "layers_info_detailed.csv"))) and\
            (os.path.isfile(os.path.join(file_path, "layers_output.csv"))) or \
            (os.path.isfile(os.path.join(file_path, "model_inference.csv"))):
        return 'Results have been generated successfully'
    else:
        return 'Some error while generating results'