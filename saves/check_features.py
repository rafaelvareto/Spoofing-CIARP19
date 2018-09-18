import numpy as np

features, labels = np.load('saves/protocol_01_train.npy')

for (feat,lab) in zip(features, labels):
    if len(feat) != 0:
        print 