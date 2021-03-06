import numpy as np

FILENAME = 'protocol_01_train.npy'
removeNAN = True

if removeNAN:
    print('Loading numpy file')
    features, labels = np.load(FILENAME)
    
    print('FEATURES', len(features))
    new_features = list()
    new_labels = list()
    for feat,lab in zip(features, labels):
        if not np.any(np.isnan(feat)):
            new_features.append(feat.tolist())
            new_labels.append(lab)
        else:
            print(lab, feat[0:5], len(feat))

    print('Saving numpy file')
    np.save(FILENAME + '_new', [new_features, new_labels])
else:
    pass
print('DONE')