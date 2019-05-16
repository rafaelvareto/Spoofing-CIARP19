import json
import numpy as np
import os


path = '.'
json_files = [file for _root, _path, _file in os.walk(path) for file in _file if file.endswith('.json')]
print(json_files)

json_files.sort()
for file in json_files:
	with open(file) as json_ptr:
		json_data = json.load(json_ptr)
		live_mean = np.mean(json_data['live'])
		spoof_mean = np.mean(json_data['spoof'])
		live_std = np.std(json_data['live'])
		spoof_std = np.std(json_data['spoof'])
		print('{:40.40}: LE {:.3f} | SE {:.3f} | MEAN {:.2f} | STD {:.2f}'.format(file, live_mean, spoof_mean, 100*(live_mean+spoof_mean)/2, 100*(live_std+spoof_std)/2))