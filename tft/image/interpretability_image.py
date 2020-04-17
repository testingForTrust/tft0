# author: TFTAuthors@accenture.com

from lime import lime_image
import os
import csv
import json
import numpy as np
import matplotlib.image as mpimg
from skimage.segmentation import mark_boundaries
from . utils_image import read_img_names_labels_csv, read_images, generate_experiments
import pandas as pd
import math
import time


class ModelInterpretation:

    """
    class that abstracts the task of auditing a model based on interpretability.
    Generates explanations using Local Interpretable Model Agnostic Explanation (LIME) method
    Paper: https://arxiv.org/pdf/1602.04938
    Source Ref: https://github.com/marcotcr/lime
    """

    def __init__(self, model, path_to_org_imgs, img_label_csv_path, path_to_save_res, project_name,
                 class_index_json_path, num_features=5, num_samples=1000, hide_rest=True):

        """
        :param model: An object of a model class created by the user. Expected to have the parameters "self.sess",
               "self.x" (where x is a input placeholder) and "self.logits", "self.image_size_height",
               "self.image_size_width" and "self.num_channels"
        :param path_to_org_imgs: a string - a local absolute path to original images
        :param img_label_csv_path: a string - a local absolute path to a csv file that contains mapping between image
               names and corresponding labels. The column names of the csv file must be “ImageName” and “Label”
               respectively.
        :param path_to_save_res: a string - a local absolute path to save results of the audit
        :param project_name: a string - represents a name under which an audit/test is performed
        :param class_index_json_path: a string - a local absolute path to a .json file that contains the index-class
               mapping [Example: https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json]
        :param num_features: number of super pixels to include in the explanation(By default, 5)
        :param num_samples: size of the neighbourhood to learn the linear model (By default, 1K as per the lime api:
               https://lime-ml.readthedocs.io/en/latest/lime.html#module-lime.lime_image)
        :param hide_rest: By default, True - makes the non-explanation part of the image black
        """
        self.model = model
        self.PATH_TO_ORG_IMAGES = path_to_org_imgs
        self.IMAGE_LABELS_CSV_PATH = img_label_csv_path
        self.PATH_TO_SAVE_RESULTS = path_to_save_res
        self.PROJECT_NAME = project_name
        self.CLASS_INDEX_JSON_PATH = class_index_json_path
        self.num_features = num_features
        self.num_samples = num_samples
        self.hide_rest = hide_rest
        self.BATCH_SIZE = 50

        # A string value used to generate experiments.csv when user runs interpretability method
        self.methodType = "explainabilityValue"

        self.experiments_filepath = os.path.join(self.PATH_TO_SAVE_RESULTS, self.PROJECT_NAME)

        # Code added to generate the settings.csv file.

        self.root_folder_to_save_settings = os.path.join(self.PATH_TO_SAVE_RESULTS, self.PROJECT_NAME,
                                                         'Interpretability')
        if not os.path.isdir(self.root_folder_to_save_settings):
            os.makedirs(self.root_folder_to_save_settings, exist_ok=True)
        # settings.csv
        with open(os.path.join(self.root_folder_to_save_settings, "settings.csv"), 'w', newline='') as output_file:
            output_writer = csv.writer(output_file, delimiter=',')
            output_writer.writerow(['Num Features', 'Num Samples', 'Hide Rest(hide background)'])
            output_writer.writerow([self.num_features, self.num_samples, self.hide_rest])
            output_writer.writerow(['number of super pixels to include in the explanation(By default: 5)',
                                    'size of the neighbourhood to learn the linear model (By default: 1K as per the '
                                    'lime api)', 'By default: True - makes the non-explanation part of the image black'])
        print("settings.csv has been generated..")

# Version for Speed Improvement

    # 1. Batch size in LIME explain_instance
    # 2. Predict method for batch of images
    # 3. limeoutput.csv append for every batch

    def run(self):
        """
        main function that does the task of generating explanations using Local Interpretable Model Agnostic Explanation
         (LIME) method.

        """
        image_height = self.model.image_size_height
        image_width = self.model.image_size_width

        # image_names: List of image names like a.JPEG, b.JPEG etc..
        # y : List of true label names for images

        image_names, y = read_img_names_labels_csv(self.IMAGE_LABELS_CSV_PATH)

        samples_path_list = [os.path.join(self.PATH_TO_ORG_IMAGES, img_name) for img_name in image_names]

        input_size = len(samples_path_list)
        batch_evaluator: int = math.ceil((input_size / self.BATCH_SIZE))
        explainer = lime_image.LimeImageExplainer()

        root_folder_to_save_explanations = os.path.join(self.PATH_TO_SAVE_RESULTS, self.PROJECT_NAME,
                                                        'Interpretability', 'Results')

        if not os.path.isdir(root_folder_to_save_explanations):
            os.makedirs(root_folder_to_save_explanations, exist_ok=True)

        # Create empty .csv file with column names for saving LIME output at every batch evaluation

        df_ = pd.DataFrame(columns=['IMAGE', 'ORIGINAL LABEL', 'Top n Predictions', 'Explanations',
                                    'Original=Predicted?', 'Probability Of Top Most Prediction',
                                    'Probability Of Top n Predictions', 'misclsfied_with_high_prob',
                                    'corr_clsfied_with_low_prob', 'corr_clsfied_with_high_prob'])
        df_.to_csv(os.path.join(root_folder_to_save_explanations, 'limeoutput.csv'), index=False)
        with open(self.CLASS_INDEX_JSON_PATH) as index_class_mapping_file:
            mapping_dictionary = json.load(index_class_mapping_file)
        start_time = time.time()
        print("START TIME", start_time)

        # Batch evaluation starts with a batch of 50 images
        for j in range(batch_evaluator):
            list_of_top_n_preds = []  # list of lists of classes
            list_of_explanation_names = []  # list of lists having explanation names

            # list of true/false to put a sample under the category of misclassified samples
            misclsfied_with_high_prob = []

            truth_eq_pred = []  # list of true/false to identify if truth was equal to prediction

            prob_top_pred = []  # list of lists having probabilities for explanations

            top_most_prediction = []  # list of predicted class names

            # list of true/false to put a sample under the category of samples classified
            corr_clsfied_with_low_prob = []
            # correctly with low probability (i.e <=0.5)

            # list of true/false to put a sample under the category of samples classified
            corr_clsfied_with_high_prob = []
            # correctly with high probability (i.e >=0.5)

            list_of_prob_top_n_preds = []  # list of lists of probabilities
            # starting index of the batch
            index_from = self.BATCH_SIZE * j
            # ending index of the batch
            index_to = input_size if batch_evaluator - j == 1 else (j * self.BATCH_SIZE + self.BATCH_SIZE)
            image_names_batch = image_names[index_from:index_to]
            y_batch = y[index_from:index_to]
            unique_label_names_batch = list(set(y_batch))
            samples_path_list_batch = samples_path_list[index_from:index_to]
            top_most_prediction_batch = []  # list of top most predicted classes for the batch of samples
            prob_top_pred_batch = []  # list of top most predicted probabilities for the batch of samples

            # Create folders by class names to save original image and corresponding explanations for predictions
            for class_names in unique_label_names_batch:
                folder_path_to_save_explanations = os.path.join(root_folder_to_save_explanations, class_names)
                if not os.path.isdir(folder_path_to_save_explanations):
                    os.makedirs(folder_path_to_save_explanations, exist_ok=True)

            # Shape (n_samples, h, w, c)
            samples_ndarray_batch = read_images(samples_path_list_batch, image_height, image_width)
            # Predictions for Batch of Images (n_samples,n_classes)
            preds = self.model.predict(samples_ndarray_batch)

            # Get explanation for each image
            for img_name, label, img_nparray, p in zip(image_names_batch, y_batch, samples_ndarray_batch, preds):

                # print("prediction array for one image -- >", p)

                img = img_nparray.astype(np.uint8)
                folder_path_to_save_explanations = os.path.join(root_folder_to_save_explanations, label,
                                                                img_name.split('.')[0])
                if not os.path.isdir(folder_path_to_save_explanations):
                    os.makedirs(folder_path_to_save_explanations, exist_ok=True)
                # save original image
                mpimg.imsave(os.path.join(folder_path_to_save_explanations, img_name), img)

                # list of top n predictions for each image samples
                top_n_preds = []
                # list of probabilities of top n predictions for each sample
                prob_top_n_preds = []
                # img_nparray.astype(np.uint8)
                explanation = explainer.explain_instance(img, self.model.predict, top_labels=3, hide_color=0,
                                                         num_samples=self.num_samples, batch_size=self.num_samples)
                # print(explanation.local_exp)

                # Get and save explanations for top n predictions

                explanation_names = []
                for i in range(1, 4):  # we will have for only top 3 explanations
                    top_predicted_label_index = p.argsort()[-i]
                    top_predicted_label_name = mapping_dictionary[str(top_predicted_label_index)]
                    top_n_preds.append(top_predicted_label_name)
                    prob_top_n_preds.append(p[top_predicted_label_index])
                    if i == 1:
                        top_most_prediction_batch.extend([top_predicted_label_name])
                        prob_top_pred_batch.append(p[top_predicted_label_index])
                    temp, mask = explanation.get_image_and_mask(top_predicted_label_index, positive_only=True,
                                                                num_features=self.num_features,
                                                                hide_rest=self.hide_rest)
                    print("temp is ", np.max(temp), np.min(temp))
                    print("mask is ", np.max(mask), np.min(mask))
                    mpimg.imsave(os.path.join(folder_path_to_save_explanations,
                                              'explaining_' + str(i) + '_' + top_predicted_label_name + '.JPEG'),
                                 mark_boundaries(temp / np.max(temp), mask))
                    explanation_names.append('explaining_' + str(i) + '_' + top_predicted_label_name + '.JPEG')

                list_of_explanation_names.append(explanation_names)
                list_of_top_n_preds.append(top_n_preds)
                list_of_prob_top_n_preds.append(prob_top_n_preds)
                top_most_prediction.extend(top_most_prediction_batch)

            print("y_batch", y_batch)
            print("top_most_prediction_batch", top_most_prediction_batch)
            truth_eq_pred_batch = [(y_batch[i].lower() == top_most_prediction_batch[i].lower())
                                   for i in range(0, len(top_most_prediction_batch))]

            for k in range(0, len(truth_eq_pred_batch)):
                if not truth_eq_pred_batch[k] and prob_top_pred_batch[k] > .50:
                    misclsfied_with_high_prob.append('True')
                else:
                    misclsfied_with_high_prob.append('False')
                if truth_eq_pred_batch[k] and prob_top_pred_batch[k] < .50:
                    corr_clsfied_with_low_prob.append('True')
                else:
                    corr_clsfied_with_low_prob.append('False')
                if truth_eq_pred_batch[k] and prob_top_pred_batch[k] > .50:
                    corr_clsfied_with_high_prob.append('True')
                else:
                    corr_clsfied_with_high_prob.append('False')
                # print(y_batch[i])

            truth_eq_pred.extend(truth_eq_pred_batch)
            prob_top_pred.extend(prob_top_pred_batch)
            print("truth_eq_pred_batch", truth_eq_pred_batch)
            print("prob_top_pred_batch", prob_top_pred_batch)
            # Batch evaluation Ends..
        # Write results to output file

            # BATCH VARIABLES

            d = {'IMAGE': image_names_batch, 'ORIGINAL LABEL': y_batch, 'Top n Predictions': list_of_top_n_preds,
                 'Explanations': list_of_explanation_names, 'Original=Predicted?': truth_eq_pred_batch,
                 'Probability Of Top Most Prediction': prob_top_pred_batch, 'Probability Of Top n Predictions':
                 list_of_prob_top_n_preds, 'misclsfied_with_high_prob': misclsfied_with_high_prob,
                 'corr_clsfied_with_low_prob': corr_clsfied_with_low_prob, 'corr_clsfied_with_high_prob':
                 corr_clsfied_with_high_prob}
            df = pd.DataFrame(d, columns=['IMAGE', 'ORIGINAL LABEL', 'Top n Predictions', 'Explanations',
                                          'Original=Predicted?', 'Probability Of Top Most Prediction',
                                          'Probability Of Top n Predictions', 'misclsfied_with_high_prob',
                                          'corr_clsfied_with_low_prob', 'corr_clsfied_with_high_prob'])

            df.to_csv(os.path.join(root_folder_to_save_explanations, 'limeoutput.csv'), mode='a', header=False,
                      index=False)

        elapsed_time = (time.time() - start_time)
        print("ELAPSED TIME", elapsed_time)

        # Generate experimets.csv file
        generate_experiments(self.experiments_filepath, self.methodType)
        # Check if the result file has been generated, sort and return message
        if os.path.isfile(os.path.join(root_folder_to_save_explanations, "limeoutput.csv")):
            data_frame = pd.read_csv(os.path.join(root_folder_to_save_explanations, 'limeoutput.csv'), header=[0])

            data_frame = data_frame.sort_values(by=['Original=Predicted?', 'Probability Of Top Most Prediction'],
                                                ascending=[True, False])

            data_frame.to_csv(os.path.join(root_folder_to_save_explanations, 'limeoutput.csv'), index=False)
            return "Results have been generated successfully"
        else:
            settings_filepath = os.path.join(self.root_folder_to_save_settings, "settings.csv")
            os.remove(settings_filepath)
            print("Settings files deleted")
            return "Some error while generating interpretability results"
