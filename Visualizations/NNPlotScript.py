#
# Visualizer Script
# @author: Philippe Wyder (PMW2125)
# @description: visualization script to plot training metrics and testing metrics
#
#
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import csv
import time
import re
from pathlib import Path
from os import listdir
from os.path import isfile, isdir, join
# include multiprocessing
from multiprocessing import Pool

RESULT_FOLDER = '/media/ron/DataSSD/models/SlenderBeamDataset/dimensionality_analysis'
sns.set(style="white")

def getDSType(file_name):
	if "test" in file_name:
		return 'test'
	elif "validation" in file_name:
		return "validation"
	elif "train" in file_name:
		return "train"
	else:
		"unknown DS Type"
def gen_metrics(eval_data, eval_file, ds_type, labels, outputs, perror):
	#print(eval_data[0:3])
	metrics_log = {}
	for lbl, out, perr in zip(labels, outputs, perror):
		np_lbls = np.array(eval_data[lbl])
		np_outs = np.array(eval_data[out])
		metrics_log[lbl + "_MAE"] = get_mae(np_lbls, np_outs)
		metrics_log[lbl + "_MAPE"] = np.mean(np.array(eval_data[perr]))
	return metrics_log

def get_mae(np_labels, np_outputs):
	nr_labels = len(np_labels)
	return np.sum(np.absolute(np_labels-np_outputs))/nr_labels

# load training CSV-File
def gen_plots(root_path):
	metrics_log_filename = root_path / 'metrics.csv'
	log_files = [f for f in listdir(root_path) if isfile(join(root_path, f)) and 'log' in f]
	evaluation_files = [f for f in listdir(root_path) if isfile(join(root_path, f)) and 'evaluation' in f]
	for log_file in log_files:
		print('processing ', log_file)
		log_path = root_path / log_file
		try:
			log_data = pd.read_csv(log_path)
		except Exception as e:
			print(e," ", log_path)
		#print(log_data[0:3])
		#print(type(log_data))
		data = pd.melt(log_data, id_vars = ['Epoch'],  value_vars = ['TotTrainLoss', 'TotValidationLoss'],
			var_name='LossType', value_name='Loss')
		#print(data[0:3])
		data_lr = pd.melt(log_data, id_vars = ['Epoch'],  value_vars = ['LearningRate'],
			var_name='LR', value_name='LearningRate')
		#print(data_lr[0:3])


		sns.set(color_codes=True)
		fig1, ax = plt.subplots(figsize=(16,9))
		ax.set(yscale = 'log')
		loss_plot = sns.lineplot(x = 'Epoch', y = 'Loss', hue = 'LossType', style = 'LossType',
								 markers=True, dashes=False, data = data, ax = ax)
		#ax2 = plt.twinx()
		#lr_plot = sns.lineplot(x = 'Epoch', y = 'LearningRate',  data = data_lr, ax=ax2, color='tab:green')
		lr_plot = loss_plot
		lr_plot.tick_params(axis='y', labelcolor='tab:green')
		lr_plot.set_title('Loss vs. Epoch')
		fig1 = lr_plot.get_figure()
		fig1.tight_layout()
		fig1.savefig(root_path / "plot_lossVsEpoch_{}.svg".format(log_file[-16:-4]))
		fig1.savefig(root_path / "plot_lossVsEpoch_{}.png".format(log_file[-16:-4]))

	#
	# Evaluation Log Plots
	#
	cummulative_log = []
	for eval_file in evaluation_files:
		print('processing ', eval_file)
		eval_path = root_path / eval_file
		#print(eval_path)
		try:
			eval_data = pd.read_csv(eval_path, encoding = 'utf-8')
		except Exception as e:
			print(e," ", eval_path)
		ds_type = getDSType(eval_file)
		#print(eval_data[0:3])
		#print(list(eval_data))
		labels = [col_name for col_name in list(eval_data) if 'lbl' in col_name]
		outputs = [col_name for col_name in list(eval_data) if 'model' in col_name]
		perror = [col_name for col_name in list(eval_data) if 'perr' in col_name]
		metrics_log = gen_metrics(eval_data, eval_file, ds_type, labels, outputs, perror)
		metrics_log['dataset'] = ds_type
		cummulative_log.append(metrics_log)
		#print(labels)
		#print(outputs)
		## add error column
		for lbl, out, perr in zip(labels, outputs, perror):
			data = pd.melt(eval_data, id_vars = ['volume'], value_vars = [lbl, out],
							var_name = 'Data', value_name = 'Value')
			#data_err = pd.melt(eval_data, id_vars = ['volume'], value_vars = [perr],
			#				var_name = 'Error', value_name = 'Percentage')
			data = data.sort_values(by='volume')
			fig1, ax = plt.subplots(figsize=(16,9))
			#ax2 = plt.twinx()
			#eval_plot = sns.lineplot(x = 'volume', y = 'Percentage', data = data_err, ax=ax2)
			eval_plot = sns.lineplot(x = 'volume', y = 'Value', hue = 'Data', data = data, ax=ax)
			eval_plot.set_title('{} and {} vs. volume on {} dataset'.format(lbl, out, ds_type))
			fig = eval_plot.get_figure()
			fig.tight_layout()
			fig.savefig(root_path / "plot_{}and{}vs.volume{}_{}.svg".format(lbl, out, ds_type, re.findall(r'\d{8}\d+', str(eval_file))[0]), transparent=True)
			fig.savefig(root_path / "plot_{}and{}vs.volume{}_{}.png".format(lbl, out, ds_type, re.findall(r'\d{8}\d+', str(eval_file))[0]), transparent=True)
		plt.close('all')
	print("Cumulative Log Length: ", len(cummulative_log))
	if len(cummulative_log):
		fieldnames = cummulative_log[0].keys()
		print("Generating ", metrics_log_filename, "for", root_path)
		with open(metrics_log_filename, 'w') as csvfile:
			writer = csv.DictWriter(csvfile, fieldnames = sorted(fieldnames))
			writer.writeheader()
			print(fieldnames, '\n', cummulative_log)
			writer.writerows(cummulative_log)
def getSubfolders(result_folder):
	print("Find subfolders in", result_folder)
	folders = [result_folder / f for f in listdir(result_folder) if isdir(join(result_folder, f))]
	subfolders = []
	for f in folders:
		for sf in listdir(f):
			if isdir(join(f,sf)):
				subfolders.append(f / sf)
	return subfolders
def main(result_folder, multifolder):
	# Expects:	result_folder <Path>
	#			multifolder <bool>	
	if result_folder is None:
		result_folder = Path(RESULT_FOLDER)
	if not multifolder:
		gen_plots(result_folder)
	else:
		folders = getSubfolders(result_folder)
		for folder in folders:
			print(folder)
		# Parallel processing
		pool = Pool() # Create a multiprocessing Pool
		start_time = time.time()
		pool.map(gen_plots, folders)  # process data_inputs iterable with pool
		print("Parallel time: %s seconds" % (time.time()-start_time))
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Neural Network Parser')
    parser.add_argument('--folder', type=str, default=RESULT_FOLDER, metavar=RESULT_FOLDER,
                        help='Provide path to directory that either contains sub-directories with NN training files or is containnig the files themselves')
    parser.add_argument('--multifolder', type=bool, default=False, metavar='True or False',
                        help='True if the directory is a super directory of folders containing NN training files to be plotted.')
    args = parser.parse_args()
    main(Path(args.folder), args.multifolder)