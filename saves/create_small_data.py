import numpy as np
import os

def big_to_small(file_name, max_feats=100000):
    print('Loading ', file_name)
    new_feature = list()
    new_label = list()
    new_path = list()

    feature_list, label_list, path_list = np.load(file_name)
    
    counter = 0
    for triplet in zip(feature_list, label_list, path_list):
        x_data, y_data, z_data = triplet[0], triplet[1], triplet[2]
        new_feature.append(x_data)
        new_label.append(y_data)
        new_path.append(z_data)
        
        counter += 1
        if counter > max_feats:
            break

    new_name = file_name.replace('.npy', '-new')
    np.save(new_name, [new_feature, new_label, new_path])


PROBE_FILE = "SiW-probe-50p.npy"
probe_dict = big_to_small(PROBE_FILE, max_feats=100000)

TRAIN_FILE = "SiW-train-50p.npy"
train_dict = big_to_small(TRAIN_FILE, max_feats=100000)
