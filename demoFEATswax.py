import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

from face_spoofing import FaceSpoofing
from myPlots import MyPlots
from sklearn.metrics import precision_recall_curve, auc

feature_file = './datasets/SWAX-dataset.npy'
samples_video = 4
threshold_value = 0


with open('./protocols_avi/Protocol-02-train.json') as infile:
	train_json = json.load(infile)
with open('./protocols_avi/Protocol-02-valid.json') as infile:
	valid_json = json.load(infile)
with open('./protocols_avi/Protocol-02-test.json') as infile:
	probe_json = json.load(infile)

if os.path.isfile(feature_file):
	feature_list, label_list, path_list = np.load(feature_file, allow_pickle=True)
	feature_list = list(feature_list)
	label_list = list(label_list)
	path_list = list(path_list)
	print(len(feature_list[0]), len(feature_list), len(label_list), len(path_list))
else:
	print('No dataset feature file found!')
	
errors_list = list()
labels_list = list()
scores_list = list()

assert(len(train_json) == len(probe_json))
for fold_idx in range(len(train_json)):
	print('>> FOLD {}'.format(fold_idx + 1))

	probe_fold_path  = [item[0] for item in probe_json[fold_idx]]
	probe_fold_label = [item[1] for item in probe_json[fold_idx]]
	valid_fold_path  = [item[0] for item in valid_json[fold_idx]]
	valid_fold_label = [item[1] for item in valid_json[fold_idx]]
	train_fold_path  = [item[0] for item in train_json[fold_idx]]
	train_fold_label = [item[1] for item in train_json[fold_idx]]

	# gather images for training stage
	train_dict = dict()
	for (img_path, img_label, img_feat) in zip(path_list, label_list, feature_list):
		if (img_label, img_path) in train_dict:
			train_dict[(img_label, img_path)].append(img_feat)
		else:
			train_dict[(img_label, img_path)] = [img_feat]

	# gather images for development stage
	valid_dict = dict()
	for (img_path, img_label, img_feat) in zip(path_list, label_list, feature_list):
		if (img_label, img_path) in valid_dict:
			valid_dict[(img_label, img_path)].append(img_feat)
		else:
			valid_dict[(img_label, img_path)] = [img_feat]

	# gather images for evaluation stage
	probe_dict = dict()
	for (img_path, img_label, img_feat) in zip(path_list, label_list, feature_list):
		if (img_label, img_path) in probe_dict:
			probe_dict[(img_label, img_path)].append(img_feat)
		else:
			probe_dict[(img_label, img_path)] = [img_feat]

	# fit the model
	classifier = FaceSpoofing()
	classifier.import_features(feature_dict=train_dict)
	classifier.trainESVM(models=30, samples4model=20, pos_label='real', neg_label='wax') 

	# THRESHOLD: Predict samples
	validation_labels = list()
	validation_scores = list()
	for (label, path) in valid_dict.keys():
		pred_label, pred_score = classifier.predict_feature(valid_dict[(label, path)])
		validation_labels.append(+1) if label == 'real' else validation_labels.append(-1)
		validation_scores.append(pred_score)
	precision, recall, threshold = precision_recall_curve(validation_labels, validation_scores)
	fmeasure = [(thr, (2 * (pre * rec) / (pre + rec))) for pre, rec, thr in zip(precision[:-1], recall[:-1], threshold)]
	fmeasure.sort(key=lambda tup:tup[1], reverse=True)
	best_threshold = fmeasure[0][0]
	print('SELECTED THRESHOLD', best_threshold)

	# keep testing results record
	evaluation_labels = list()
	evaluation_scores = list()
	counter_dict = {img_label:0.0 for img_label in label_list}
	mistake_dict = {img_label:0.0 for img_label in label_list}
	for (label, path) in probe_dict.keys():
		pred_label, pred_score = classifier.predict_feature(probe_dict[(label, path)])
		evaluation_labels.append(+1) if label == 'real' else evaluation_labels.append(-1)
		evaluation_scores.append(pred_score)
		if pred_label != label: mistake_dict[label] += 1
		counter_dict[label] += 1
	# Keep record of APCER, BPCER and ROC
	errors_list.append({label:(mistake_dict[label]/counter_dict[label]) for label in counter_dict.keys()})
	labels_list.append(evaluation_labels)
	scores_list.append(evaluation_scores)
	print(len(evaluation_labels), len(evaluation_scores))

	# save data to disk 
	with open('./score/esvm_swax_acer.json','w') as outfile:
		json.dump(errors_list, outfile) 
	with open('./score/esvm_swax_roc.json','w') as outfile:
		json.dump({'labels':[label for label in labels_list], 'scores':[score for score in scores_list]}, outfile) 
	# generate ROC curve
	plt.figure()
	roc_data = MyPlots.merge_roc_curves(labels_list, scores_list, name='ROC Average')
	MyPlots.plt_roc_curves([roc_data,])
	plt.savefig('./score/esvm_swax_roc.pdf')
	plt.close()