# author: TFTAuthors@accenture.com
# Purpose: This module abstracts different methods to audit word embeddings for Robustness
# ref: https://github.com/vecto-ai/word-benchmarks for sample test data

from tft.text.test_data_text import read_test_data_analogy, read_test_data_cluster, read_test_data_toefl_synonyms, read_test_data_outlier
import numpy as np
import csv
import pickle
from sklearn.cluster import KMeans
# from tft.utils_text import  cluster_labels_to_decoupled_clusters
import tft.text.utils_text as utils

class Cluster:
    def __init__(self, model):
        self.model = model
        self.embedding_normalized = self.model.embedding_dict_normalized  # model class must contain a variable called embedding_normalized

    def run(self):
        # user calls this run method

        # TODO: assert the keys & values of embedding
        # get the test data & call cluster_test Method
        cluster_data_list, cluster_labels_list, num_cluster_list = read_test_data_cluster()

        total_correct = 0
        embedding_predictions = []  # save afterwards
        correct_or_not = []  # save afterwards

        for i in range(len(cluster_data_list)):
            clusters, correct = self.clustering_test(cluster_data_list[i], cluster_labels_list[i], num_cluster_list[i])
            embedding_predictions.append(clusters)
            correct_or_not.append(correct)
            if correct == 1:
                total_correct += 1
            if i % 1000 == 0:
                print(i)

        print("total correct are ", total_correct)
        utils.cluster_labels_to_decoupled_clusters(cluster_data_list, cluster_labels_list, embedding_predictions,
                                             correct_or_not)
        # uncomment below lines for saving the results to disk
        # pickle.dump(embedding_predictions,open('clustering_predictions_list.pkl','wb'))
        # pickle.dump(correct_or_not,open('clustering_correct_or_not_list.pkl','wb'))
        utils.overall_metric_save_clustering(total_correct,len(cluster_data_list))

    def clustering_test(self, wordlist, labels_list, num_clusters):
        word_emb = []
        for i in range(len(wordlist)):
            if wordlist[i] not in self.embedding_normalized:
                return [-1], -1
            else:
                word_emb.append(self.embedding_normalized[wordlist[i]])  # /np.linalg.norm(embedding[wordlist[i]]))
        kmeans = KMeans(n_clusters=num_clusters, random_state=17).fit(np.array(word_emb))
        clusters = kmeans.labels_
        clusters_dict = {}
        clusters_words_dict = {}
        for i in range(len(clusters)):
            if clusters[i] in clusters_dict.keys():
                clusters_dict[clusters[i]].append(labels_list[i])  # ' , '+wordlist[i]
            else:
                clusters_dict[clusters[i]] = [labels_list[i]]
        # print()
        for i in range(num_clusters):
            for j in range(len(clusters_dict[i]) - 1):
                if clusters_dict[i][j] != clusters_dict[i][j + 1]:
                    return clusters, 0
        return clusters, 1


class AnalogyReasoning:

    def __init__(self, model):
        self.model = model
        self.embedding_normalized = self.model.embedding_dict_normalized  # model class must contain a variable called embedding_dict

    def analogy_test(self, w1, w2, w3):
        # print("inside analogy_test")

        if w1 not in self.embedding_normalized or w2 not in self.embedding_normalized or w3 not in self.embedding_normalized:
            return [w1, w2, w3, 'word is not present'], -10

        w1_emb = self.embedding_normalized[w1]
        w2_emb = self.embedding_normalized[w2]
        w3_emb = self.embedding_normalized[w3]
        target_emb = w2_emb + w3_emb - w1_emb
        target_emb /= np.linalg.norm(target_emb)
        max = -10
        target_word = None
        for word in self.embedding_normalized:  # same as self.embedding_normalized.keys()
            if word in [w1, w2, w3]:  # don't consider words in w1,w2,w3
                continue
            emb = self.embedding_normalized[word]
            cosine_sim = np.dot(emb, target_emb)
            if cosine_sim > max:
                max = cosine_sim
                target_word = word
        return [w1, w2, w3, target_word], max

    def run(self):

        # read the test_data
        (csv_row_data, f_data), analogy_subtype_set, analogy_subtype_number_dict = read_test_data_analogy()

        analogy_subtype_correct_dict = {}
        analogy_subtype_percentage_correct_dict = {}
        for ele in analogy_subtype_set:
            analogy_subtype_correct_dict[ele] = 0
        analogy_subtype_correct_dict['total'] = 0
        analogy_subtype_correct_dict['not_present'] = 0

        correct_or_not = []  # a list of 1,0,-1 for each analogy being correct, incorrect or not-present respectively.
        row_num = 0
        predictions = []
        analogy_data_list_for_saving = []
        for row in csv_row_data:
            if row_num != 0:  # except the first header row 0
                pred, score = self.analogy_test(row[2], row[3], row[4])
                # print("inside loop")
                predictions.append((pred, score))
                if pred[3].lower() == row[5]:
                    analogy_subtype_correct_dict[row[1]] += 1
                    analogy_subtype_correct_dict['total'] += 1
                    correct_or_not.append(1)
                    iscorrect = 1
                elif pred[3] == 'word is not present':
                    analogy_subtype_correct_dict['not_present'] += 1
                    correct_or_not.append(-1)
                    iscorrect = -1
                else:
                    correct_or_not.append(0)
                    iscorrect = 0
                analogy_data_list_for_saving.append([row[2], row[3], row[4], row[5], pred[3], iscorrect])
            row_num += 1
            if row_num % 1000 == 999:
                print(row_num)
                # break #comment/remove this line

        f_data.close() # closing the google_analogies.csv file
        utils.save_analogy_results(analogy_data_list_for_saving)

        f_concept = open('concept_metric_analogy.csv', 'w', newline='')
        csvw_concept = csv.writer(f_concept, delimiter=',')
        csvw_concept.writerow(['concept', 'correct', 'total', 'accuracy'])

        f_total = open('overall_metric_analogy.csv', 'w', newline='')
        csvw_total = csv.writer(f_total, delimiter=',')
        csvw_total.writerow(['correct', 'total', 'accuracy'])

        f_not_present = open('not_present.csv', 'w', newline='')
        csvw_not_present = csv.writer(f_not_present, delimiter=',')
        csvw_not_present.writerow(['not_present', 'total', 'ratio'])


        for subtype in analogy_subtype_number_dict:
            analogy_subtype_percentage_correct_dict[subtype] = analogy_subtype_correct_dict[subtype] / analogy_subtype_number_dict[
                subtype]
            if subtype !='total':
                csvw_concept.writerow([subtype,analogy_subtype_correct_dict[subtype], analogy_subtype_number_dict[subtype],
                               analogy_subtype_percentage_correct_dict[subtype]])
            else:
                csvw_total.writerow([analogy_subtype_correct_dict[subtype], analogy_subtype_number_dict[subtype],
                               analogy_subtype_percentage_correct_dict[subtype]])

        csvw_not_present.writerow([analogy_subtype_correct_dict['not_present'], analogy_subtype_number_dict['total'],
                                   analogy_subtype_correct_dict['not_present']/analogy_subtype_number_dict['total']])

        f_concept.close()
        f_total.close()
        f_not_present.close()
        print(analogy_subtype_percentage_correct_dict)
        print(predictions[:10])

        # uncomment below lines for saving the results to disk
        # pickle.dump(analogy_subtype_percentage_correct_dict,open('analogy_subtype_percentage_correct_dict.pkl','wb'))
        # pickle.dump(analogy_subtype_correct_dict,open('analogy_subtype_correct_dict.pkl','wb'))
        # pickle.dump(predictions,open('analogy_predictions_list.pkl','wb'))
        # pickle.dump(correct_or_not,open('analogy_correct_or_not_list.pkl','wb'))


class SynonymToefl:

    def __init__(self, model):
        self.model = model
        self.embedding_normalized = self.model.embedding_dict_normalized  # model class must contain a variable called embedding_dict

    def run(self):

        questions,answers = read_test_data_toefl_synonyms()
        predicted_answers = []
        predicted_answers_indexes = []
        correct = 0
        correct_or_not = []

        for i in range(len(answers)):

            ques_index = 5 * i
            word = questions[ques_index]
            word_emb_norm = self.embedding_normalized[word]
            max = 0

            for j in range(4):
                other_word = questions[ques_index + 1 + j]
                if other_word in self.embedding_normalized:
                    other_word_emb_norm = self.embedding_normalized[other_word]
                    cosine = np.dot(word_emb_norm, other_word_emb_norm)
                    if cosine > max:
                        max = cosine
                        pred_ans = other_word
                        pred_index = j + 97

            predicted_answers.append(pred_ans)
            predicted_answers_indexes.append(pred_index)

            if (ord(answers[i]) == predicted_answers_indexes[i]):
                correct += 1
                correct_or_not.append(1)
            else:
                correct_or_not.append(0)
        print("\naccuracy on toefl is %.2f percentage\n" % ((correct * 100) / 80))

        # uncomment below lines for saving the results to disk
        # pickle.dump(predicted_answers,open('toefl_predicted_answers_list.pkl','wb'))
        # pickle.dump(correct_or_not,open('toefl_correct_or_not_list.pkl','wb'))
        # pickle.dump(predicted_answers_indexes, open('toefl_predicted_answers_index_list.pkl', 'wb'))

class Outlier:
    def __init__(self, model):
        self.model = model
        self.embedding = self.model.embedding_dict  # only Outlier class needs un-normalized embeddings

    def run(self):
        outlier_list = read_test_data_outlier()
        total_correct = 0
        total_not_present = 0
        total_incorrect = 0

        correct_or_not = []  # save afterwards
        outlier_indexes = []  # save afterwards

        for i in range(len(outlier_list)):
            correct, outlier_index = self.outlier_test(outlier_list[i])
            correct_or_not.append(correct)
            outlier_indexes.append(outlier_index)
            if correct == 1:
                total_correct += 1
            elif correct == 0:
                total_incorrect += 1
            elif correct == -1:
                total_not_present += 1

        print("total correct,incorrect, not-present are ", total_correct, total_incorrect, total_not_present)

        # uncomment below lines for saving the results to disk
        # pickle.dump(outlier_indexes,open('outlier_predicted_indexes_list.pkl','wb'))
        # pickle.dump(correct_or_not,open('outlier_correct_or_not_list.pkl','wb'))


    def outlier_test(self,wordlist):

        word_emb = []
        score = []  # l2 score for every word as outlier
        for i in range(len(wordlist)):
            if wordlist[i] not in self.embedding:
                return -1, -1
            else:
                word_emb.append(self.embedding[wordlist[i]])


        for i in range(len(wordlist)):
            mem_list = list(word_emb)
            outlier = mem_list.pop(i)
            mem_centroid = np.mean(mem_list, axis=0)
            score.append(np.linalg.norm(outlier - mem_centroid))

        outlier_index = np.argmax(score)

        if outlier_index == len(wordlist) - 1:
            return 1, outlier_index
        else:
            return 0, outlier_index
