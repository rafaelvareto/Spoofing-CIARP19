import argparse
import numpy as np
import pickle
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import auc

class MyPlots():
    @classmethod
    def compute_fscore(self, precision, recall, threshold):
        pre = [precision[idx] for idx in range(0, len(precision)-1)]
        rec = [recall[idx] for idx in range(0, len(recall)-1)]
        fscores = [2 * (pr * re) / (pr + re) for (pr,re) in zip(precision, recall)]
        complete_zip = zip(fscores, threshold)
        complete_zip.sort(key=lambda tup: tup[0], reverse=True)
        return (complete_zip[0][0], complete_zip[0][1])

    @classmethod
    def get_thesholds(self, scores):
        thresh_size = 1000.0 - len(scores)
        min_score, max_score = min(scores), max(scores)
        step = (max_score - min_score) / thresh_size
        thresh = np.arange(max_score + (1.2 * step), min_score - (1.2 * step), -step).tolist()
        thresh.extend(scores)
        thresh.sort(reverse=True)
        return thresh
        
    @classmethod
    def merge_cmc_curves(self, cmc_scores, name='CMC'):
        y_axis = np.mean(cmc_scores, axis=0)
        x_axis = range(1, len(y_axis) + 1)
        aucs = [auc(x_axis, score) for score in cmc_scores]
        # averaging values
        cmc_values = dict()
        cmc_values['name'] = name
        cmc_values['y_axis'] = y_axis
        cmc_values['x_axis'] = x_axis
        cmc_values['avg'] = np.mean(aucs, axis=0) / (x_axis[-1] - x_axis[0])
        cmc_values['std'] = np.std(aucs, axis=0)
        return cmc_values

    @classmethod
    def merge_det_curves(self, labels_list, scores_list, name='DET'):
        false_negative_rates = list()
        false_positive_rates = list()
        # compute values manually
        for (labels, scores) in zip(labels_list, scores_list):
            threshold = MyPlots.get_thesholds(scores)
            temp_false_neg = list()
            temp_false_pos = list()
            for thresh in threshold:
                # computing false negative rate: FNR = 1 - TPR, TPR = TP/P
                tpos = sum(float(la > 0) for (sc,la) in zip(scores, labels) if sc >= thresh)
                pos = sum(float(la > 0) for la in labels)
                temp_false_neg.append(1 - (tpos / pos))
                # computing false positive rate: FAR = FPR = FP/(FP+TN) = FP/N
                fpos = sum(float(la < 0) for (sc,la) in zip(scores, labels) if sc >= thresh)
                neg = sum(float(la < 0) for la in labels)
                temp_false_pos.append(fpos / neg)
            false_negative_rates.append(temp_false_neg)
            false_positive_rates.append(temp_false_pos)
        aucs = [auc(false_pos, false_neg) for (false_pos, false_neg) in zip(false_positive_rates, false_negative_rates)]
        # averaging values
        det_values = dict()
        det_values['name'] = name
        det_values['y_axis'] = np.mean(false_negative_rates, axis=0)
        det_values['x_axis'] = np.mean(false_positive_rates, axis=0)
        det_values['avg'] = np.mean(aucs, axis=0)
        det_values['std'] = np.std(aucs, axis=0)
        return det_values

    @classmethod
    def merge_dir_curves(self, labels_list, scores_list, people_list, person_list, rank_id=1, name='DIR'):
        true__positive_rates = list()
        false_positive_rates = list()
        detect_identfy_rates = list()
        # compute values manually
        for (labels, scores, people, person) in zip(labels_list, scores_list, people_list, person_list):
            threshold = MyPlots.get_thesholds(scores)
            temp_true__pos = list()
            temp_false_pos = list()
            temp_detec_idf = list()
            total = sum([+1 for item in labels if item == True])
            for thresh in threshold:
                # computing false positive rate: FPR = FAR = FP/(FP+TN) = FP/N
                fpos = sum(float(la < 0) for (sc,la) in zip(scores, labels) if sc >= thresh)
                neg = sum(float(la < 0) for la in labels)
                temp_false_pos.append(fpos / neg)
                # computing detection and identification rate: DIR
                correct = sum([+1.0 if (labels[inner] == True) and (scores[inner] >= thresh) and (person[inner] in people[inner][:rank_id]) else 0.0 for inner in range(len(labels))])
                temp_detec_idf.append(correct)
            temp_detec_idf = np.divide(temp_detec_idf, total)
            true__positive_rates.append(temp_true__pos)
            false_positive_rates.append(temp_false_pos)
            detect_identfy_rates.append(temp_detec_idf)
        aucs = [auc(false_positive, detect_id) for (false_positive, detect_id) in zip(false_positive_rates, detect_identfy_rates)]
        # averaging values
        dir_values = dict()
        dir_values['name'] = name
        dir_values['y_axis'] = np.mean(detect_identfy_rates, axis=0)
        dir_values['x_axis'] = np.mean(false_positive_rates, axis=0)
        dir_values['avg'] = np.mean(aucs, axis=0)
        dir_values['std'] = np.std(aucs, axis=0)
        return dir_values

    @classmethod 
    def merge_prc_curves(self, labels_list, scores_list, name='PRC'): 
        recall_rates = list() 
        precis_rates = list() 
        # compute values manually 
        for (labels, scores) in zip(labels_list, scores_list): 
            threshold = MyPlots.get_thesholds(scores) 
            temp_recall = list() 
            temp_precis = list() 
            for thresh in threshold: 
                # computing recall: TPR = TAR = TP/(FN+TP) = TP/P 
                tpos = sum(float(la > 0) for (sc,la) in zip(scores, labels) if sc >= thresh) 
                pos = sum(float(la > 0) for la in labels) 
                temp_recall.append(tpos / pos) 
                # computing precision: PPV = TP/(FP+TP) 
                tpos = sum(float(la > 0) for (sc,la) in zip(scores, labels) if sc >= thresh) 
                fpos = sum(float(la < 0) for (sc,la) in zip(scores, labels) if sc >= thresh) 
                temp_precis.append(tpos / (tpos + fpos)) if (tpos + fpos) > 0 else temp_precis.append(0)  
            recall_rates.append(temp_recall) 
            precis_rates.append(temp_precis)
        aucs = [auc(rec, pre) for (rec, pre) in zip(recall_rates, precis_rates)]
        # averaging values
        prc_values = dict()
        prc_values['name'] = name
        prc_values['y_axis'] = np.mean(precis_rates, axis=0)
        prc_values['x_axis'] = np.mean(recall_rates, axis=0)
        prc_values['avg'] = np.mean(aucs, axis=0)
        prc_values['std'] = np.std(aucs, axis=0)
        return prc_values

    @classmethod
    def merge_roc_curves(self, labels_list, scores_list, name='ROC'):
        true__positive_rates = list()
        false_positive_rates = list()
        # compute values manually
        for (labels, scores) in zip(labels_list, scores_list):
            threshold = MyPlots.get_thesholds(scores)
            temp_true__pos = list()
            temp_false_pos = list()
            for thresh in threshold:
                # computing true positive rate: TPR = TAR = TP/(FN+TP) = TP/P
                tpos = sum(float(la > 0) for (sc,la) in zip(scores, labels) if sc >= thresh)
                pos = sum(float(la > 0) for la in labels)
                temp_true__pos.append(tpos / pos)
                # computing false positive rate: FPR = FAR = FP/(FP+TN) = FP/N
                fpos = sum(float(la < 0) for (sc,la) in zip(scores, labels) if sc >= thresh)
                neg = sum(float(la < 0) for la in labels)
                temp_false_pos.append(fpos / neg)
            true__positive_rates.append(temp_true__pos)
            false_positive_rates.append(temp_false_pos)
        aucs = [auc(false_positive, true_positive) for (false_positive, true_positive) in zip(false_positive_rates, true__positive_rates)]
        # averaging values
        roc_values = dict()
        roc_values['name'] = name
        roc_values['y_axis'] = np.mean(true__positive_rates, axis=0)
        roc_values['x_axis'] = np.mean(false_positive_rates, axis=0)
        roc_values['avg'] = np.mean(aucs, axis=0)
        roc_values['std'] = np.std(aucs, axis=0)
        return roc_values

    @classmethod
    def plt_cmc_curves(self, scores):
        for score in scores:
            plt.plot(score['x_axis'], score['y_axis'], label='%s (R1@ %0.2f)' % (score['name'], score['y_axis'][0])) # color='blue', linestyle='--'
        plt.xticks(fontsize=7)
        plt.yticks(fontsize=7)
        plt.xlim([1, len(score['x_axis'])])
        plt.ylim([0.0, 1.01])
        plt.xlabel('Rank', fontsize=7)
        plt.ylabel('Identification Rate', fontsize=7)
        plt.title('Cumulative Matching Characteristic', fontsize=8)
        plt.legend(loc="lower right", fontsize=7)
        plt.grid()

    @classmethod
    def plt_det_curves(self, scores, name='DET'):
        for score in scores:
            area = auc(score['x_axis'], score['y_axis'])
            plt.plot(score['x_axis'], score['y_axis'], label='%s (area: %0.2f±%0.2f)' % (score['name'], score['avg'], score['std']))
        plt.xticks(fontsize=7)
        plt.yticks(fontsize=7)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.01])
        plt.xlabel('False Positive Rate', fontsize=7)
        plt.ylabel('False Negative Rate', fontsize=7)
        plt.title('Detection Error Trade-off', fontsize=8)
        plt.legend(loc="upper right", fontsize=7)
        plt.grid()

    @classmethod
    def plt_dir_curves(self, scores):
        for score in scores:
            area = auc(score['x_axis'], score['y_axis'])
            plt.plot(score['x_axis'], score['y_axis'], label='%s (area: %0.2f±%0.2f)' % (score['name'], score['avg'], score['std']))
        plt.xticks(fontsize=7)
        plt.yticks(fontsize=7)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.01])
        plt.xlabel('False Acceptance Rate', fontsize=7)
        plt.ylabel('Detection and Identification Rate', fontsize=7)
        plt.title('Open-set Receiver Operating Characteristic', fontsize=8)
        plt.legend(loc="lower right", fontsize=7)
        plt.grid()

    @classmethod
    def plt_prc_curves(self, scores):
        for score in scores:
            plt.plot(score['x_axis'], score['y_axis'], label='%s (area: %0.2f±%0.2f)' % (score['name'], score['avg'], score['std']))
        plt.xticks(fontsize=7)
        plt.yticks(fontsize=7)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.01])
        plt.xlabel('Recall', fontsize=7)
        plt.ylabel('Precision', fontsize=7)
        plt.title('Precision-Recall Curve', fontsize=8)
        plt.legend(loc="lower left", fontsize=7)
        plt.grid()

    @classmethod
    def plt_roc_curves(self, scores):
        for score in scores:
            area = auc(score['x_axis'], score['y_axis'])
            plt.plot(score['x_axis'], score['y_axis'], label='%s (area: %0.2f±%0.2f)' % (score['name'], score['avg'], score['std']))
        plt.xticks(fontsize=7)
        plt.yticks(fontsize=7)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.01])
        plt.xlabel('False Positive Rate', fontsize=7)
        plt.ylabel('True Positive Rate', fontsize=7)
        plt.title('Receiver Operating Characteristic', fontsize=8)
        plt.legend(loc="lower right", fontsize=7)
        plt.grid()


def main():
    methodAWS = dict()
    methodOUR = dict()
    with open('./plots/demoSC/demoAWS-SC.file', 'rb') as infile:
        methodAWS['cumuls'], methodAWS['labels'], methodAWS['scores'], methodAWS['people'], methodAWS['person'] = pickle.load(infile, encoding='latin1')
    with open('./plots/demoSC/demoSC-50-20-5-0-True-True.file', 'rb') as infile:
        methodOUR['cumuls'], methodOUR['labels'], methodOUR['scores'], methodOUR['people'], methodOUR['person'] = pickle.load(infile, encoding='latin1')

    print('AWS')
    awsSScmc = MyPlots.merge_cmc_curves(methodAWS['cumuls'], name='AWS')
    awsSSdet = MyPlots.merge_det_curves(methodAWS['labels'], methodAWS['scores'], name='AWS')
    awsSSdir = MyPlots.merge_dir_curves(methodAWS['labels'], methodAWS['scores'], methodAWS['people'], methodAWS['person'], name='AWS')
    awsSSprc = MyPlots.merge_prc_curves(methodAWS['labels'], methodAWS['scores'], name='AWS')
    awsSSroc = MyPlots.merge_roc_curves(methodAWS['labels'], methodAWS['scores'], name='AWS')

    print('OURS')
    ourSScmc = MyPlots.merge_cmc_curves(methodOUR['cumuls'], name='OURS')
    ourSSdet = MyPlots.merge_det_curves(methodOUR['labels'], methodOUR['scores'], name='OURS')
    ourSSdir = MyPlots.merge_dir_curves(methodOUR['labels'], methodOUR['scores'], methodOUR['people'], methodOUR['person'], name='OURS')
    ourSSprc = MyPlots.merge_prc_curves(methodOUR['labels'], methodOUR['scores'], name='OURS')
    ourSSroc = MyPlots.merge_roc_curves(methodOUR['labels'], methodOUR['scores'], name='OURS')

    plt.figure()
    MyPlots.plt_cmc_curves([awsSScmc, ourSScmc])
    plt.savefig('./plots/SCcompCMC.png')
    plt.close() 

    plt.figure()
    MyPlots.plt_det_curves([awsSSdet, ourSSdet])
    plt.savefig('./plots/SCcompDET.png')
    plt.close() 

    plt.figure()
    MyPlots.plt_dir_curves([awsSSdir, ourSSdir])
    plt.savefig('./plots/SCcompDIR.png')
    plt.close() 
    
    plt.figure()
    MyPlots.plt_prc_curves([awsSSprc, ourSSprc])
    plt.savefig('./plots/SCcompPRC.png')
    plt.close() 

    plt.figure()
    MyPlots.plt_roc_curves([awsSSroc, ourSSroc])
    plt.savefig('./plots/SCcompROC.png')
    plt.close()   


if __name__ == "__main__":
    main()