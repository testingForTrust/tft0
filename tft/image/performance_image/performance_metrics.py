# author: TFTAuthors@accenture.com
# Purpose: This file defines a class that helps determine the performance of a model
# [a] time to inference
# [b] time it takes to pre_process a batch of data by the model
# [c] time to load the model
# [d] time spent on each layer by the model


import time
import os
from tft.image.utils_image import read_img_names_labels_csv, read_images, result_message_performance_metrics
import tensorflow as tf
from csv import DictWriter, reader
import pandas as pd
from tensorflow.python.framework.errors_impl import InvalidArgumentError, FailedPreconditionError


class PerformanceMetrics:
    def __init__(self, model, input_placeholder_tensor_name, path_to_model, model_file, load_model_file_flag,
                 path_to_save_results, project_name, image_samples_folder, image_vs_labels_csv, image_height,
                 image_width, num_samples_batch=64):
        """

        :param model: An object of a model class created by the user. Expected to have the parameters "self.sess",
        "self.x" and "self.logits", "self.image_size_height", "self.image_size_width" and "self.num_channels"
        :param input_placeholder_tensor_name: the placeholder that takes a batch of input images
        :param path_to_model: a string - local absolute path to the model
        :param model_file: a string - model file name like .meta, .h5 or .pb
        :param load_model_file_flag : a boolean flag that indicates whether the load_model function in the user defined
         Model class contains "path_to_model"
        :param path_to_save_results: a string - absolute path to save results
        :param project_name: a string - represents a name under which the performance metrics is evaluated
        :param image_samples_folder : a string - absolute path to a folder that contains sample images
        :param image_vs_labels_csv : A string - a local absolute path to a csv file that contains mapping between image
               names and corresponding labels. The column names of the csv file must be “ImageName” and “Label”
               respectively.
        :param image_height: height of the images that the model is trained to take
        :param image_width: width of the images that the model is trained to take
        :param num_samples_batch : batch size on which performance metrics is evaluated. At most 64.
        """
        self.model = model
        self.input_placeholder_tensor_name = input_placeholder_tensor_name
        self.path_to_model = path_to_model
        self.model_file = model_file
        self.load_model_file_flag = load_model_file_flag
        self.path_to_save_results = path_to_save_results
        self.project_name = project_name
        self.image_samples_folder = image_samples_folder
        self.image_vs_labels_csv = image_vs_labels_csv
        self.image_height = image_height
        self.image_width = image_width
        self.num_samples_batch = num_samples_batch
        self.time_inference = 0  # time it takes to read a batch of data by the model and give output
        self.time_pre_process = 0  # time it takes to pre_process a batch of data by the model
        self.time_load_model = 0  # time it takes to load a model from scratch.
        self.root_folder_to_save_results = os.path.join(self.path_to_save_results, self.project_name)

    def metric_inference(self, image_batch):
        """

        :param image_batch: batch of nd arrays of shape (n_samples, h, w, c)
        :return: float. Time in seconds - Inference time for a batch of samples
        """
        assert (image_batch.shape == (self.num_samples_batch, self.image_height, self.image_width, 3)), \
            "the batch shape must be (n_samples, h, w, c)"
        tic = time.time()
        self.model.predict(image_batch)
        toc = time.time()
        self.time_inference = toc-tic
        print("Inference time for a batch of '%s' images is " % (len(image_batch)) + str(self.time_inference) + " s")
        return self.time_inference

    def metric_pre_process_time(self, image_batch):
        """

        :param image_batch: batch of nd arrays of shape (n_samples, h, w, c)
        :return: float. Time in seconds - time it takes to pre process a batch of images
        """
        assert (image_batch.shape == (self.num_samples_batch, self.image_height, self.image_width, 3)), \
            "the batch shape must be (n_samples, h, w, c)"
        tic = time.time()
        self.model.pre_process(image_batch)
        toc = time.time()
        self.time_pre_process = toc-tic
        print("Time taken to pre_process a batch of '%s' images is " % (len(image_batch)) + str(self.time_pre_process) +
              " s")
        return self.time_pre_process

    def metric_load_model(self):
        # load_model is to be called with in a new variable scope. If not, we get an error as the model would already
        # be loaded and have a reference inside "model" object.
        # tf.AUTO_REUSE creates variables if they do not exist, and return them otherwise;
        # For more info: https://www.tensorflow.org/api_docs/python/tf/variable_scope
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            if self.load_model_file_flag:
                tic = time.time()
                self.model.load_model(self.path_to_model, self.model_file)
                toc = time.time()
                self.time_load_model = toc-tic
            else:
                tic = time.time()
                self.model.load_model(self.path_to_model)
                toc = time.time()
                self.time_load_model = toc-tic
            print("Time taken to load the model is " + str(self.time_load_model) +
                  " s")
            # assert sess != self.model.sess, "EQUAL!"
        return self.time_load_model

    def metric_layers(self, image_batch, input_tensor_name):
        """

        :param image_batch: batch of nd arrays of shape (n_samples, h, w, c)
        :param input_tensor_name: the input placeholder name
        :return: a boolean indicating success or failure
        """
        print("Generating layers_info_detailed.csv...")
        # First get all operations in the graph from the session
        model_graph = self.model.sess.graph
        all_operations_in_graph = model_graph.get_operations()  # list of all operations in the graph
        input_placeholder_tensor = model_graph.get_tensor_by_name(input_tensor_name)  # "Placeholder:0"
        assert len(input_placeholder_tensor.shape) == 4, "The input tensor shape must be of length 4. The first " \
                                                         "dimension being 'number of samples in the batch', second " \
                                                         "being 'height of each image', third being 'width of each " \
                                                         "image' and the last being 'number of channels'"
        X = self.model.pre_process(image_batch)
        first_row = True
        for an_op in all_operations_in_graph:
            # Ignore saver operations, placeholder & Assign operations and non fetchable operations in the graph
            if "save" not in an_op.name.lower() and "placeholder" not in an_op.name.lower() and "Assign_" not in \
                    an_op.name.lower() and model_graph.is_fetchable(an_op):
                # Type of op is  <class 'tensorflow.python.framework.ops.Operation'>
                # op = self.model.sess.graph.get_operation_by_name(an_op.name)  # op.name is a string

                # get the output tensors of this operation
                op_outputs = an_op.outputs  # The list of tensor objects representing the outputs of this op
            # for an_output in op_outputs:
                # print("The operation name ", an_op.name)
                # print("output Tensors ", op_outputs)
                try:
                    tic = time.time()
                    self.model.sess.run(op_outputs, feed_dict={input_placeholder_tensor: X})
                    toc = time.time()
                except InvalidArgumentError:
                    continue
                except FailedPreconditionError:
                    continue
                # print("Time to run in seconds-", (toc-tic))

                # open a csv file in append mode, write a row, close
                with open(os.path.join(self.root_folder_to_save_results, "layers_info_detailed.csv"), "a", newline='') \
                        as layers_info_detailed_file:
                    headers = ["OperationName", "Num Of Output Tensors This Op Produces", "Time in Seconds"]

                    layers_info_writer = DictWriter(layers_info_detailed_file, delimiter=',', fieldnames=headers)
                    if first_row:
                        layers_info_writer.writeheader()
                        first_row = False
                    layers_info_writer.writerow({"OperationName": an_op.name, "Num Of Output Tensors This Op Produces":
                                                len(op_outputs), "Time in Seconds": toc-tic})
        print("layers_info_detailed.csv generated successfully")

        # Code to get tokenised Operation Names from detailed CSV

        if os.path.isfile(os.path.join(self.root_folder_to_save_results, "layers_info_detailed.csv")):
            # read the csv
            df = pd.read_csv(os.path.join(self.root_folder_to_save_results, "layers_info_detailed.csv"))
            list_of_ops = list(df['OperationName'])
            tokens = []
            # Getting initial common sub-strings by comparing two consecutive Operation Names
            for i in range(len(list_of_ops) - 1):
                str_1 = list_of_ops[i]
                str_2 = list_of_ops[i + 1]
                if str_2 in str_1:
                    tokens.append(str_2)
            # print(len(tokens))
            # dataframe for saving sub-string & average time taken by operations starting with the sub-string
            df3 = pd.DataFrame(columns=['Sub-string in Operation Name',
                                        'Average Time taken by operations starting with Sub-string'])

            # Getting all the unique Operation Names other than sub-strings
            m = list_of_ops[1:]
            for sub_str in tokens:
                criterion = df['OperationName'].map(lambda x: x.startswith(sub_str))
                df2 = df[criterion]
                # saving average time simultaneously for the initial tokens
                df3 = df3.append({'Sub-string in Operation Name': sub_str,
                                  'Average Time taken by operations starting with Sub-string': df2['Time in Seconds'].mean()},
                                 ignore_index=True)
                list_of_ops_sub_str = df2['OperationName'].tolist()
                m = [x for x in m if x not in list_of_ops_sub_str]

            # Applying same logic on additional operations obtained to get average time taken
            for sub_str in m:
                criterion = df['OperationName'].map(lambda x: x.startswith(sub_str))
                df2 = df[criterion]
                df3 = df3.append({'Sub-string in Operation Name': sub_str,
                                  'Average Time taken by operations starting with Sub-string': df2['Time in Seconds'].mean()},
                                 ignore_index=True)
            # finally saving dataframe to CSV
            df3.to_csv(os.path.join(self.root_folder_to_save_results, "layers_output_tokened.csv"), index=False)
            print("layers_output_tokened.csv generated successfully")
            res = True
        else:
            res = False

        return res

    def compute(self):
        """
        A function that computes the model's inference metrics & layer related (tensorflow operation) metrics (basically
        time in seconds)

        :return: a bool - indicates whether results were generated successfully or not
        """
        first_row = True
        assert (self.num_samples_batch <= 64), "Batch sizes must be small, <= 64"
        image_names_list, labels = read_img_names_labels_csv(self.image_vs_labels_csv)
        if len(image_names_list) < 64:
            self.num_samples_batch = len(image_names_list)
        samples_path_list = [os.path.join(self.image_samples_folder, img_name)
                             for img_name in image_names_list[0:self.num_samples_batch]]
        batch_ndarray = read_images(samples_path_list, self.image_height, self.image_width)
        model_load_time = self.metric_load_model()
        pre_process_time = self.metric_pre_process_time(batch_ndarray)
        inf_time = self.metric_inference(batch_ndarray)
        col1 = "Time taken to load the model(in sec)"
        col2 = "Time taken to pre_process a batch of " + str(self.num_samples_batch) + " images(in sec)"
        col3 = "Time taken to predict a batch of " + str(self.num_samples_batch) + " images(in sec)"

        if not os.path.isdir(self.root_folder_to_save_results):
            os.makedirs(self.root_folder_to_save_results, exist_ok=True)
        with open(os.path.join(self.root_folder_to_save_results, "model_inference.csv"), "w", newline='') as \
                model_inference_file:
            headers = [col1, col2, col3]
            model_inference_writer = DictWriter(model_inference_file, delimiter=',', fieldnames=headers)
            if first_row:
                model_inference_writer.writeheader()
                first_row = False
            model_inference_writer.writerow({col1: model_load_time, col2: pre_process_time, col3: inf_time})
        print("model_inference.csv file generated successfully..")

        self.metric_layers(batch_ndarray, self.input_placeholder_tensor_name)
        # Check if the metrics result files have been generated successfully and give a success/failure message
        return result_message_performance_metrics(self.root_folder_to_save_results)
