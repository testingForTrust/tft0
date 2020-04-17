# author: TFTAuthors@accenture.com
# Purpose: This Module contains different utility methods for text_data
import json
import csv


def cluster_labels_to_decoupled_clusters(cluster_data_list,cluster_labels_list, embedding_predictions,correct_or_not):
    cluster_dict_list = []
    f = open('decoupled_clusters.csv', 'w', newline='')
    csvw = csv.writer(f, delimiter=',')
    csvw.writerow(['concept1','cluster_list1','concept2','cluster_list2','concept3','cluster_list3','concept4',
                   'cluster_list4','concept5','cluster_list5','cluster_number1','predicted_cluster_num1','cluster_number2',
                   'predicted_cluster_num2','cluster_number3','predicted_cluster_num3',
                   'cluster_number4','predicted_cluster_num4','cluster_number5','predicted_cluster_num5','correct_or_not'])

    for i in range(len(cluster_data_list)):

        wordlist,clusters_orig,clusters_preds = cluster_data_list[i],cluster_labels_list[i], embedding_predictions[i]

        clusters_dict_orig = {}
        clusters_dict_preds = {}
        # clusters_dict_preds['correct'] = correct_or_not[i]
        for j in range(len(clusters_orig)):
            if str(clusters_orig[j]) in clusters_dict_orig.keys():
                clusters_dict_orig[str(clusters_orig[j])].append(wordlist[j])  # ' , '+wordlist[i]
            else:
                clusters_dict_orig[str(clusters_orig[j])] = [wordlist[j]]

        for j in range(len(clusters_preds)):
            if str(clusters_preds[j]) in clusters_dict_preds.keys():
                clusters_dict_preds[str(clusters_preds[j])].append(wordlist[j])  # ' , '+wordlist[i]
            else:
                clusters_dict_preds[str(clusters_preds[j])] = [wordlist[j]]


        cluster_dict_list.append([clusters_dict_orig,clusters_dict_preds])
        # csvw.writerow([clusters_dict_orig,clusters_dict_preds])

        csvlist = []

        for key in clusters_dict_orig.keys():
            csvlist.append(key)
            csvlist.append(clusters_dict_orig[key])

        csvlist.extend(['']*(5-len(clusters_dict_orig))*2)

        for key in clusters_dict_preds.keys():
            csvlist.append(key)
            csvlist.append(clusters_dict_preds[key])

        csvlist.extend(['']*(5-len(clusters_dict_preds))*2)
        csvlist.append(correct_or_not[i])

        if len(clusters_dict_orig) > 5:
            print("\n\n\n\n THIS SHOULD NOT HAVE HAPPEN \n\n\n\n\n")
            exit()
        csvw.writerow(csvlist)


def overall_metric_save_clustering(correct,total):
    f = open('overall_metric_clustering.csv', 'w', newline='')
    csvw = csv.writer(f, delimiter=',')
    csvw.writerow(['correct','total','accuracy'])
    csvw.writerow([correct,total,correct/total])
    f.close()

def save_analogy_results(analogy_list):
    f = open('saved_analogy_results.csv', 'w', newline='',encoding='utf-8')
    csvw = csv.writer(f, delimiter=',')
    csvw.writerow(['word1', 'word2', 'word3','word4','prediction','correct_or_not'])

    for list in analogy_list:
        csvw.writerow(list)
    f.close()
