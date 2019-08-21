import json
import numpy as np

with open('epls_swax_acer.json') as infile: #'esvm_swax_acer.json'
	acer_data = json.load(infile)
	real_error = [item['real'] for item in acer_data]
	wax_error  = [item['wax'] for item in acer_data]
	print('APCER {}\\pm{}'.format(round(np.mean(wax_error),2) , round(np.std(wax_error),2) ))
	print('BPCER {}\\pm{}'.format(round(np.mean(real_error),2), round(np.std(real_error),2)))
	mean_error = (np.mean(wax_error) + np.mean(real_error))/2
	mean_std   =  (np.std(wax_error) +  np.std(real_error))/2
	print(' ACER {}\\pm{}'.format(round(mean_error,2), round(mean_std,2)))