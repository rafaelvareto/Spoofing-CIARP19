import numpy as np
import os

def big_to_small(file_name, max_feats=100000):
    print('Loading ', file_name)
    counter_dict = dict()
    new_feature = list()
    new_label = list()
    new_path = list()

    feature_list, label_list, path_list = np.load(file_name)
    
    for triplet in zip(feature_list, label_list, path_list):
        x_data, y_data, z_data = triplet[0], triplet[1], triplet[2]

        if y_data in counter_dict:
            counter_dict[y_data] += 1
        else:
            counter_dict[y_data] = 1

        if counter_dict[y_data] < max_feats:
            new_feature.append(x_data)
            new_label.append(y_data)
            new_path.append(z_data)

    new_name = file_name.replace('.npy', '-new')
    np.save(new_name, [new_feature, new_label, new_path])


PROBE_FILE = "SiW-probe.npy"
probe_dict = big_to_small(PROBE_FILE, max_feats=30000)

TRAIN_FILE = "SiW-train.npy"
train_dict = big_to_small(TRAIN_FILE, max_feats=30000)
