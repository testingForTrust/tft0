# authors:TFTAuthors
# This file is prepared to test word embeddings

import numpy as np
import os
from tft.text.robustness_text import AnalogyReasoning, Cluster, SynonymToefl, Outlier


class Model:

    def __init__(self, model_path):
        """
        load word embedding
        :param model_path: absolute path to the model
        """
		# implementation to read a text file & create a embedding dictionary
        self.embedding_dict = {}
        self.embedding_dict_normalized = {}
        
        with open(model_path, 'rb') as f:
            for l in f:
                line = l.decode().split()
                self.embedding_dict[line[0]] = np.array(line[1:]).astype(np.float)
                self.embedding_dict_normalized[line[0]] = self.embedding_dict[line[0]]/np.linalg.norm(self.embedding_dict[line[0]])


PATH_TO_MODEL = r'C:\Users\Downloads\glove\glove.6B.300d.txt'

# Handle custom data?
# We can supply the data used by us where users can test using it or may test with their custom data
# PATH_TO_TEST_DATA = 'google-analogies-lowercase.csv'
model = Model(PATH_TO_MODEL)

task1 = AnalogyReasoning(model)  #===> works as expected
task1.run()                       #===> works as expected

task2 = Cluster(model)  #===> works as expected
task2.run()             #===> works as expected

task3 = SynonymToefl(model)  #===> works as expected
task3.run()                  #===> works as expected

task4 = Outlier(model)	 #===> works as expected
task4.run()  #===> works as expected


