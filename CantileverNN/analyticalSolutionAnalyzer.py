#
#  Total Deflection Analyzer (Analytical Method)
#  @author: Philippe Wyder (PMW2125)
#  @description: Analyzes beam dataset and computes total deflection from Moment Of Inertia
#  
from FEABeamDataset import *
from NNEvaluator import NNEvaluator
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import chain
import pickle
PLOT_DATA = True
SHOW_PLOTS = False
SAFE_PLOTS = True
WRITE_TO_FILE = False
WRITE_ANALYSIS_LOG = True
SEPARATE_MULTILABELS = False
ANALYSIS_RANGE = [0,17500]
#root_path = Path('/home/ron/GeneratedData/data/TestSetRectCantilever')

# Slender Beam Data
#dataset_name = "SlenderBeamDS"
#root_path = Path('/media/ron/DataSSD/datasets/SlenderBeamData/SlenderBeamDataTF2000/')
#EF123_Model = Path('/media/ron/DataSSD/models/Pre_2020NOV03/SlenderBeamDataset/ConvNetExtended/ef123/BS100.1e-05/best_model_ef123img_agrayscale_128SlenderBeamDataTF2000202004081506.ckpt')
#TotDisp_Model = Path('/media/ron/DataSSD/models/Pre_2020NOV03/SlenderBeamDataset/ConvNetExtended/LblPerformanceTable/TotDisp/BS100.0.0001/best_model_TotDispimg_agrayscale_128SlenderBeamDataTF2000202004070708.ckpt')

# Twisted Beam Models
dataset_name = "TwistedBeamDS"
root_path = Path('/media/ron/DataSSD/datasets/TwistedBeamDS/')
EF123_Model = Path('/media/ron/DataSSD/models/Pre_2020NOV03/TwistedBeamDataset/ConvNetExtended/ef123/BS100.0001/best_model_ef123img_agrayscale_128TwistedBeamDS202004082002.ckpt')
TotDisp_Model = Path('/media/ron/DataSSD/models/Pre_2020NOV03/TwistedBeamDataset/ConvNetExtended/LblPerformanceTable/TotDisp/BS100.0.0001/best_model_TotDispimg_agrayscale_128TwistedBeamDS202003101823.ckpt')

# instantiate evaluators on GPU0 and GPU1 for EF123 and TotDisp
EF123_Evaluator = NNEvaluator(EF123_Model, 0, neuralnetwork = 'ConvNetExtended', img_res = 128, num_img_layers = 1, num_classes = 3)  
TotDisp_Evaluator = NNEvaluator(TotDisp_Model, 1, neuralnetwork = 'ConvNet', img_res = 128, num_img_layers = 1, num_classes = 1) 

# write to file
outfile_name = 'principal_vs_nonprincipal.csv'
to_file_dic = [] 

# SET FIGURE FONT SIZE
sns.set(font_scale=1.25)
#set mathfont size
mpl.rcParams['mathtext.fontset'] = 'stix'

# SET Vibration Roots
beta_n = np.array([	0.00312517344785326861074218040179702360428,
					0.00782348522162362429406065296336635341565,
					0.0130912623970626876081016809712742840964,
					0.0183259012247924449844455818464245048994
				])
zeta_n = np.array([	1.87510406871196116644530824107821416257011173353107,
					4.69409113297417457643639177801981204938989673754577,
					7.85475743823761256486100858276457045784854192923005,
					10.9955407348754669906673491078547029396129727746516
				])
'''
Maximum Reaction Force

at the fixed end can be expressed as:

RA = F                              (1a)

where

RA = reaction force in A (N, lb)

F = single acting force in B (N, lb)
Maximum Moment

at the fixed end can be expressed as

Mmax = MA

          = - F L                               (1b)

where

MA = maximum moment in A (N.m, N.mm, lb.in)

L = length of beam (m, mm, in)
Maximum Deflection

at the end of the cantilever beam can be expressed as

δB = F L3 / (3 E I)                                      (1c)

where

δB = maximum deflection in B (m, mm, in)

E = modulus of elasticity (N/m2 (Pa), N/mm2, lb/in2 (psi))

I = moment of Inertia (m4, mm4, in4)
'''
class analyticalCantileverbeam:
	def __init__(self, loadcase = 'left_fixed_right_end_point_load', secondMoment = None,
					force = 0, elasticity = None, density = None, beam_length = 600, volume = None):
		self.loadcase = type
		self.secondMoment = secondMoment
		self.F = force*1000 # force [N] = force*1000 [kg*mm*s^(-2)]
		self.E = elasticity
		self.rho = density
		self.L = beam_length
		self.Volume = volume
	def getDeflection(self, bending_axis = 0):
		#Due to the force vector [Fx, 0, 0], displacement in the y- and z-direction are 0
		max_del_p, I_err = self.getMaxDeflection()
		max_del = self.F[bending_axis]*np.power(self.L, 3)/(3*self.E*self.secondMoment[bending_axis])
		#to_file_dic.append({'max_del': max_del, 'max_del_p':max_del_p, 'I_err':I_err})
		#print("max_del_p-max_del", max_del_p-max_del)
		return max_del_p
		#return np.power(self.F*self.L, 3)/(3*self.E*self.secondMoment)
	def getPrincipalAxeswithAngle(self):
		theta_p = (1.0/2.0)*np.arctan((2*self.secondMoment[2])/(self.secondMoment[1]-self.secondMoment[0]))
		Ixx_p = self.secondMoment[0]*np.power(np.cos(theta_p),2)+self.secondMoment[1]*np.power(np.sin(theta_p),2)-self.secondMoment[2]*np.sin(2*theta_p) 
		Iyy_p = self.secondMoment[0]*np.power(np.sin(theta_p),2)+self.secondMoment[1]*np.power(np.cos(theta_p),2)+self.secondMoment[2]*np.sin(2*theta_p)
		return (Ixx_p, Iyy_p, theta_p)
	def getMaxDeflection(self):
		Ixx_p, Iyy_p, theta_p = self.getPrincipalAxeswithAngle()
		#print("see if sum of Ixx + Iyy is the same: ", self.secondMoment[0]+self.secondMoment[1]-Ixx_p-Iyy_p) 
		r_theta = np.array([[np.cos(theta_p), -np.sin(theta_p), 0.0],
							[np.sin(theta_p), np.cos(theta_p) , 0.0],
							[0.0			, 0.0			  , 1  ]])
		F_p = r_theta.dot(self.F)
		del_p = F_p*np.power(self.L, 3)/(3*self.E*self.secondMoment)
		return (np.linalg.norm(del_p), self.secondMoment[0]+self.secondMoment[1]-Ixx_p-Iyy_p)

	def getEigenfrequency(self, n, axis = 0):
		Ixx_p, Iyy_p, theta_p = self.getPrincipalAxeswithAngle()
		secondMoment_p = [Ixx_p, Iyy_p]
		n = n-1 # account for 0-indexing
		# Pre-Computed Roots of cosh(beta_n*L)*cos(beta_n*L) + 1 = 0 in ascending order (where L = 600)
		linear_mass_density = self.rho*(self.Volume/self.L) # rho*crosssection_area = linear_mass_density
		#w_n = np.power(beta_n[n],2)*np.sqrt( (self.E*self.secondMoment[axis])/ linear_mass_density) # first formulation
		w_n = np.power(zeta_n[n],2)/np.power(self.L,2)*np.sqrt( (self.E*self.secondMoment[axis]) / linear_mass_density ) # MIT Formulation P10 src:https://ocw.mit.edu/courses/mechanical-engineering/2-002-mechanics-and-materials-ii-spring-2004/labs/lab_1_s04.pdf
		w_n_p = np.power(zeta_n[n],2)/np.power(self.L,2)*np.sqrt( (self.E*secondMoment_p[axis]) / linear_mass_density ) # MIT Formulation P10 src:https://ocw.mit.edu/courses/mechanical-engineering/2-002-mechanics-and-materials-ii-spring-2004/labs/lab_1_s04.pdf 
		to_file_dic.append({'w_n': w_n, 'w_n_p': w_n_p, 'I_err':self.secondMoment[0]+self.secondMoment[1]-Ixx_p-Iyy_p})
		return w_n_p/(2*np.pi) # f_n = w_n/(2pi)
	def getEigenfrequencies(self, nr_eigenfrequencies = 3):
		# returns multiple eigenfrequencies in array (starting from the smallest)
		assert nr_eigenfrequencies < 4
		w_n_x = [self.getEigenfrequency(i+1, axis = 0).item() for i in range(nr_eigenfrequencies)]
		w_n_y = [self.getEigenfrequency(i+1, axis = 1).item() for i in range(nr_eigenfrequencies)] 
		w_n = sorted(w_n_x + w_n_y)

		return np.array(w_n[:nr_eigenfrequencies]) 


def analyze_datapoint(i, mode=None):
	assert mode is not None
	beam_path = Path(root_path / str(i))
	#print(beam_path)
	#MOI = np.load(beam_path / 'label' / 'MatrixOfInertia.npy') # ? (depending on the unit selected in FreeCAD i.e. should be mm^4)
	secondMoment = np.load(beam_path / 'label' / 'secondMoment.npy') # ? (depending on the unit selected in FreeCAD i.e. should be mm^4)
	length = np.load(beam_path / 'numbers' / 'extrude_length.npy') # mm
	Force = np.load(beam_path / 'numbers' / 'FEAloadCase.npy') # N
	volume = np.load(beam_path / 'label' / 'volume.npy')
	youngs_modulus = 7e7 # [N/mm^2] (Aluminium COMSOL E=70GPa=7e10Pa) note: 1Pa = 0.001 kg/(m*s^2)
	density = 2.7e-6 # 2.7×10^(-6) kg/mm^3 (kilograms per millimeter cubed) or 2700[kg/m^3] (Aluminium COMSOL)

	mybeam = analyticalCantileverbeam(secondMoment = secondMoment, force = Force, elasticity = youngs_modulus,
										density = density, beam_length = length, volume = volume)

	if mode == 'deflection':
		totDispAnalytical = mybeam.getDeflection(bending_axis = 0)
		#print(totDispAnalytical)
		if WRITE_TO_FILE:
			np.save(beam_path / 'label' / 'totalDisplacementAnalyticalmm.npy', totDispAnalytical)	
		# GET FEA totDisp
		totDispFEA = np.load(beam_path / 'label' / 'totalDisplacementmm.npy') # mm
		# GET Model totDisp 
		totDisp_predicted = TotDisp_Evaluator.load_and_evaluate(beam_path / 'img' / 'img_agrayscale_128.jpg') 
		#print('Deflection: {}mm FEA vs. {}mm Analytical'.format(totDispFEA, totDispAnalytical.item()))
		# MUST BE A TUPLE FOR STRUCTURED NP ARRAY CONVERSION
		return (i,	totDispFEA.item(), totDispAnalytical.item(), totDisp_predicted.item(), volume.item(),
					dp_AErr(totDispFEA.item(), totDispAnalytical.item()),
					dp_APErr(totDispFEA.item(), totDispAnalytical.item()))
	if mode == 'eigenfrequency':
		analytical_ef123 = mybeam.getEigenfrequencies(3)
		#print(analytical_ef123)
		if WRITE_TO_FILE:
			np.save(beam_path / 'label' / 'ef123AnalyticalHz.npy', analytical_ef123)
		# GET FEA eigenfrequencies	
		ef1 = np.load(beam_path / 'label' / 'ef1.npy')[0] # Hz
		ef2 = np.load(beam_path / 'label' / 'ef2.npy')[0] # Hz
		ef3 = np.load(beam_path / 'label' / 'ef3.npy')[0] # Hz
		ef123FEA = np.array([float(ef1.item()), float(ef2.item()), float(ef3.item())])
		# GET Model Eigenfrequencies
		ef123_predicted = EF123_Evaluator.load_and_evaluate(beam_path / 'img' / 'img_agrayscale_128.jpg') 
		#print('ef1: {}Hz FEA vs. {}Hz Analytical'.format(ef1.item(), analytical_ef123[0]))
		# MUST BE A TUPLE FOR STRUCTURED NP ARRAY CONVERSION
		#print("I: {}\tA:{}\tE:{}\trho:{}\tLMD:{}".format(secondMoment, volume/length, youngs_modulus, density, (volume/length)*density))
		#from IPython import embed; embed()
		return (i,	ef123FEA, analytical_ef123, torch.squeeze(ef123_predicted).numpy(), volume.item(),
					dp_AErr(ef123FEA[0].item() , analytical_ef123[0].item()),
					dp_APErr(ef123FEA[0].item(), analytical_ef123[0].item()))
def dp_Err(lbl, pred):
	# calculates error over all labels
	return float(np.mean(np.array(lbl, dtype=np.float)-np.array(pred, dtype=np.float)))
def dp_AErr(lbl, pred):
	# calculates absolute error over all labels
	return float(np.mean(abs(np.array(lbl, dtype=np.float)-np.array(pred, dtype=np.float))))
def dp_APErr(lbl, pred):
	# calculates absolute percentage error over all labels
	return float(np.mean(abs(np.array(lbl, dtype=np.float)-np.array(pred, dtype=np.float))/np.array(lbl, dtype=np.float)*100))
def write_analysis_log(data, log_file):
	fieldnames = data.dtype.names
	str_fieldnames = ','.join(fieldnames)
	pd.DataFrame(data).to_csv(log_file, index=None)
def vibrationScatterPlot(data):
	from IPython import embed; embed()
	data = np.sort(data, order='Volume')
	plt_volume = [dp[4] for dp in data]
	nr_labels = (np.array(data[0][1])).size
	if SEPARATE_MULTILABELS:
		nr_graphs = nr_labels
		iterator_range = zip(range(nr_labels), range(nr_graphs))
	else:
		nr_graphs = 1
		iterator_range = zip(range(nr_labels), np.zeros(nr_labels, dtype=int))

	print("nr_labels",nr_labels, "nr_graphs", nr_graphs)
	fig, axs = plt.subplots(nrows=1, ncols=nr_graphs, figsize=(5, 5), squeeze=False)
	'''
	plt_error = [dp_AErr(dp[1],dp[2]) for dp in data]
	axs.plot(plt_volume, plt_error, color='r', label='error')
	'''
	for lbl_idx, grph_idx in iterator_range:
		plt_FEA = [dp[1][lbl_idx] for dp in data] 
		plt_analytical = [dp[2][lbl_idx] for dp in data]
		plt_predicted = [dp[3][lbl_idx] for dp in data]
		plt_AErr = [dp_AErr(dp[1][lbl_idx],dp[2][lbl_idx]) for dp in data] 
		plt_volume = [i for i,dp in enumerate(data)]   
		axs[0][grph_idx].scatter(plt_volume, plt_FEA, label='FEA ef{}'.format(lbl_idx+1), s=10)
		axs[0][grph_idx].scatter(plt_volume, plt_analytical, label='Analytical ef{}'.format(lbl_idx+1), s=10)
		axs[0][grph_idx].scatter(plt_volume, plt_predicted, label='Predicted ef{}'.format(lbl_idx+1), s=10)
		axs[0][grph_idx].scatter(plt_volume, plt_AErr, label='Absolute Error ef{}'.format(lbl_idx+1), s=10)
		axs[0][grph_idx].set_ylabel("ef{} [Hz]".format(lbl_idx+1))
		axs[0][grph_idx].set_xlabel("Beam Volume [mm^3]")
		axs[0][grph_idx].set_title("Analytical vs. FEA ef{}".format(lbl_idx+1))
		max_y_val = 1000 
		min_y_val = 0
		range_size = max_y_val + abs(min_y_val) 
		label_step_size = range_size/25
		axs[0][grph_idx].set_yticks(np.arange(0, range_size, label_step_size))	
		axs[0][grph_idx].set_yticklabels(np.arange(min_y_val, max_y_val, label_step_size))	
	fig.set_size_inches(12,9)
	plt.legend()
	if SHOW_PLOTS:
		plt.show()
def deflectionScatterPlot(data):
	data = np.sort(data, order='Volume')
	fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(5, 5), squeeze=True)
	plt_error = [dp_Err(dp[1],dp[2]) for dp in data]
	plt_volume = [dp[4] for dp in data] 
	axs.plot(plt_volume, plt_error, color='r', label='error')
	plt_FEA = [dp[1] for dp in data] 
	plt_analytical = [dp[2] for dp in data]
	axs.scatter(plt_volume, plt_FEA, label='FEA Deflection ')
	axs.scatter(plt_volume, plt_analytical, label='Analytical Deflection')
	axs.set_ylabel("deflection in [mm]")
	axs.set_xlabel("Beam Volume [mm^3]")
	axs.set_title("Analytical Beam Deflection vs. FEA Beam Deflection")
	fig.set_size_inches(12,9)	
	plt.legend()
	if SHOW_PLOTS:
		plt.show()
def boxplot(data):
	df = pd.DataFrame(data)
	# melt data so the solving method becomes a factor variable
	df = pd.melt(df.drop(['AErr','APErr'], axis=1), id_vars=['idx','Volume'], var_name = 'method', value_name='Total Displacement')
	ax = sns.boxplot(x="method", y="Total Displacement", data=df, linewidth=2.5)	
def boxplot_error(data, title =''):
	# Seaborn Boxplot Absolute Error vs. Absolute Percentage Error
	df = pd.DataFrame(data).drop(['AErr','APErr'], axis = 1)
	df['MAE_FEA_Analytical'] = [dp_AErr(dp['FEA'], dp['Analytical']) for i,dp in df.iterrows()]
	df['AErr_FEA_Predicted'] = [dp_AErr(dp['FEA'], dp['Predicted']) for i,dp in df.iterrows()]
	df_error = pd.melt(df.drop(['FEA', 'Analytical','Predicted'], axis=1),
						id_vars=['idx','Volume'], var_name = 'comparison', value_name='error')
	df = pd.DataFrame(data).drop(['AErr','APErr'], axis = 1)
	df['MAPE_FEA_Analytical'] = [dp_APErr(dp['FEA'], dp['Analytical']) for i,dp in df.iterrows()]
	df['APErr_FEA_Predicted'] = [dp_APErr(dp['FEA'], dp['Predicted']) for i,dp in df.iterrows()]
	df_perror = pd.melt(df.drop(['FEA', 'Analytical','Predicted'], axis=1),
						id_vars=['idx','Volume'], var_name = 'comparison', value_name='error')
	#from IPython import embed; embed()
	fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(5, 5), squeeze=True)
	axs[0] = sns.boxplot(x="comparison", y="error", data=df_error, linewidth=2.5, ax = axs[0])  
	axs[1] = sns.boxplot(x="comparison", y="error", data=df_perror, linewidth=2.5, ax = axs[1])
	if SHOW_PLOTS:
		plt.show()

def barplot_error(data, title = '', label = '', units = ''):
	# Seaborn Barplot Absolute Error vs. Absolute Percentage Error
	model_type_name = ' '

	# Compute MAE values
	df0 = pd.DataFrame(data).drop(['AErr','APErr'], axis = 1)
	df0['Analytical '] = [dp_AErr(dp['FEA'], dp['Analytical']) for i,dp in df0.iterrows()]
	df0['ConvNetExtended'] = [dp_AErr(dp['FEA'], dp['Predicted']) for i,dp in df0.iterrows()]
	df_error = pd.melt(df0.drop(['FEA', 'Analytical','Predicted'], axis=1),
						id_vars=['idx','Volume'], var_name = model_type_name, value_name='error')
	# Compute MAPE Values
	df1 = pd.DataFrame(data).drop(['AErr','APErr'], axis = 1)
	df1['Analytical '] = [dp_APErr(dp['FEA'], dp['Analytical']) for i,dp in df1.iterrows()]
	df1['ConvNetExtended'] = [dp_APErr(dp['FEA'], dp['Predicted']) for i,dp in df1.iterrows()]
	df_perror = pd.melt(df1.drop(['FEA', 'Analytical','Predicted'], axis=1),
						id_vars=['idx','Volume'], var_name = model_type_name, value_name='error')
	fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(5, 5), squeeze=True)
	fig.suptitle(title, fontsize=22)
	axs[0] = sns.barplot(x=model_type_name, y="error", data=df_error, linewidth=2.5, ax = axs[0])  
	axs[0].set_ylabel("Mean Absolute Error [{}]".format(units))
	axs[1] = sns.barplot(x=model_type_name, y="error", data=df_perror, linewidth=2.5, ax = axs[1])
	axs[1].set_ylabel("Mean Absolute Percentage Error [%]")
	fig.set_size_inches(12,9)
	if SAFE_PLOTS:
		#pickle figure for future modification
		with open('{}_{}.pickle'.format(title, label), 'wb') as f: # should be 'wb' rather than 'w'
			pickle.dump(fig, f)
		plt.savefig(root_path / '{}_{}_{}.png'.format(dataset_name, title, label), dpi = 128)
	if SHOW_PLOTS:
		plt.show()

def deflection_analysis():
	log_file = root_path / 'analyticalDeflection.csv'
	result_data = []
	for i in range(ANALYSIS_RANGE[0],ANALYSIS_RANGE[1]):
		result = analyze_datapoint(i, mode = 'deflection') 
		result_data.append(result)
		#print(np.array(result))
	my_dtype = [('idx', int), ('FEA', float), ('Analytical', float), ('Predicted', float), ('Volume', float),
				('AErr', float), ('APErr', float)]
	#print(my_dtype)
	#for row in result_data:
	#	print('{}\t{}\t{}\t{}\t{}\t{}'.format(*row)) 
	data = np.array(result_data, dtype = my_dtype)
	#print(data)
	if WRITE_ANALYSIS_LOG:
		write_analysis_log(data, log_file)
	if PLOT_DATA:
		title = '{} Error Comparison \nAnalytical vs. Model Predicted Total Displacement'.format(dataset_name)
		#boxplot_error(data)
		barplot_error(data, title, 'Total Displacement', 'mm')
		#deflectionScatterPlot(data)
def vibration_analysis():
	log_file = root_path / 'analyticalVibration.csv'
	result_data = []
	for i in range(ANALYSIS_RANGE[0],ANALYSIS_RANGE[1]):
		result = analyze_datapoint(i, mode='eigenfrequency') 
		result_data.append(result)
		#print(np.array(result))
	my_dtype = [('idx', int), ('FEA', object), ('Analytical', object), ('Predicted', object), ('Volume', float),
				('AErr', float), ('APErr', float)]
	#print(my_dtype)
	#for row in result_data:
		#print('{}\t{}\t{}\t{}\t{}\t{}'.format(*row)) 
	data = np.array(result_data, dtype = my_dtype)
	#print(data)
	if WRITE_ANALYSIS_LOG:
		write_analysis_log(data, log_file)
	if PLOT_DATA:
		title = '{} Error Comparison \nAnalytical vs. Model Predicted '.format(dataset_name)
		math_notation = r'$[\mathit{f}_1, \mathit{f}_2, \mathit{f}_3]$'
		title = title + math_notation
		#boxplot_error(data)
		barplot_error(data, title, 'EF123', 'Hz')
		#vibrationScatterPlot(data)			
def main():
	deflection_analysis()
	vibration_analysis()
	with open(outfile_name, 'w') as csvfile:
		csv_columns = to_file_dic[0].keys()
		writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
		writer.writeheader()
		writer.writerows(to_file_dic)

if __name__ == '__main__':
	main()