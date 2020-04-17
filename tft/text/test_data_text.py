# author: TFTAuthors@accenture.com
# Purpose: this file has modules(functions) to load test data for testing word embeddings with respect to analogies,
# clustering, synonymns and outlier tasks

import pickle
import csv


def read_test_data_cluster():
    with open(r'.\tft\text\test_data_for_text\clustering_data\final_cluster_data_list.pkl', 'rb') as f_data:
        cluster_data_list = pickle.load(f_data)

    with open(r'.\tft\text\test_data_for_text\clustering_data\final_cluster_labels_list.pkl', 'rb') as f_labels:
        cluster_labels_list = pickle.load(f_labels)

    with open(r'.\tft\text\test_data_for_text\clustering_data\clustering_data_num_clusters_list.pkl', 'rb') as f_num:
        num_cluster_list = pickle.load(f_num)

    return cluster_data_list, cluster_labels_list, num_cluster_list


def read_test_data_analogy():
    f_data = open(r'.\tft\text\test_data_for_text\analogies_data\google-analogies-lowercase.csv', 'r')
    f_ang_sub = open(r'.\tft\text\test_data_for_text\analogies_data\analogy_subtypes.pkl', 'rb')
    f_ang_sub_num = open(r'.\tft\text\test_data_for_text\analogies_data\analogy_subtype_numbers.pkl', 'rb')
    csv_r = csv.reader(f_data, delimiter=',')
    ang_sub_set = pickle.load(f_ang_sub)
    ang_sub_num_dict = pickle.load(f_ang_sub_num)

    f_ang_sub.close()
    f_ang_sub_num.close()
    return (csv_r, f_data), ang_sub_set, ang_sub_num_dict


def read_test_data_toefl_synonyms():
    questions = pickle.load(open(r'.\tft\text\test_data_for_text\synonym_toefl_data\questionAnswers\ques_f.pkl', 'rb'))
    answers = pickle.load(open(r'.\tft\text\test_data_for_text\synonym_toefl_data\questionAnswers\ans_f.pkl', 'rb'))
    return questions, answers


def read_test_data_outlier():
    with open(r'.\tft\text\test_data_for_text\outlier_data\outlier_list.pkl', 'rb') as f_data:
        outlier_data_list = pickle.load(f_data)

    return outlier_data_list
