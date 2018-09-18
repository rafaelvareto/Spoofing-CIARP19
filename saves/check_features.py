import numpy as np

FILENAME = 'protocol_01_train.npy'

features, labels = np.load(FILENAME)

print('FEATURES', len(features))
new_features = list()
new_labels = list()
for (idx,(feat,lab)) in enumerate(zip(features, labels)):
    if not np.any(np.isnan(feat)):
        new_features.append(feat)
        new_labels.append(lab)
    else:
        print(idx, lab, feat[0:5], len(feat))

np.save(FILENAME + '_new', [new_features, new_labels])
        
print('DONE')