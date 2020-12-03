"""
# @Name: NNEvaluator (class)
# @Description: Provides a simple class to evaluate a single picture given a model and an image
# @Author: Philippe Wyder (PMW2125)
# """
import torch
import importlib
from skimage import io, transform
class NNEvaluator:
	def __init__(self, model_name, GPU, neuralnetwork = 'ConvNet', img_res = 128, num_img_layers = 1, num_classes = None):
		# set GPU to run model on
		gpu_id = 'cuda:' + str(GPU)
		self.device = torch.device(gpu_id if torch.cuda.is_available() else 'cpu')
		# load model dictionary to GPU
		self.model_dict = torch.load(model_name, map_location=self.device )
		self.model_state_dict = self.model_dict['model_state_dict']
		# Load the model checkpoint
		if "NeuralNetwork" in self.model_dict["hyperparam_dict"]:
			my_module = importlib.import_module(self.model_dict["hyperparam_dict"]["NeuralNetwork"])
		else:
			my_module = importlib.import_module(neuralnetwork)
		ConvNet = getattr(my_module, "ConvNet")
		self.model = ConvNet(num_classes = num_classes, num_img_layers = num_img_layers, img_res = img_res).to(self.device)
		#self.model.to(self.device)
		self.model.load_state_dict(self.model_state_dict)
		self.epsilon = torch.tensor(1e-12, dtype = torch.float)
	def evaluate(self, img):
		self.model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
		with torch.no_grad():
			img = img.to(self.device)
			out = self.model(img)
			out = out.to('cpu')
		return out
	def load_and_evaluate(self, img_path):
		# Loads image, performs transpose usually done in dataloader, and passes image to dataset
		image = io.imread(img_path, as_gray=True)
		image_t = image.transpose((0, 1))
		img_dim = image_t.shape
		img = torch.from_numpy(image_t).float().reshape(1,img_dim[0],img_dim[1])
		img = torch.unsqueeze(img, 0)
		return self.evaluate(img)
def main():
	# test script for all class functions
	from pathlib import Path
	import numpy as np
	from analyticalSolutionAnalyzer import analyticalCantileverbeam
	dataset_path = Path('/media/ron/DataSSD/datasets/TwistedBeamDS/')
	#beam_number = np.random.randint(14000, 17500)
	beam_number = 14572
	for i in range(3):
		model_path = Path('/home/ron/Documents/PhD/Altair/CantileverNN/FEA/TwistedBeamDS/ConvNetExtended/ef123/BS100.0.0001/best_model_ef123img_agrayscale_128TwistedBeamDS20200408200{}.ckpt'.format(i))
		my_eval = NNEvaluator(model_path, GPU=0, neuralnetwork = 'ConvNetExtended', img_res = 128, num_img_layers = 1, num_classes = 3)
		print("TEST SCRIPT FOR NNEvaluator Class: BEAM #{}".format(beam_number))
		print("LOAD AND EVALUATE on model {}".format(model_path.name))
		print(my_eval.load_and_evaluate(dataset_path / str(beam_number) / "img" / "img_agrayscale_128.jpg"))
		# Analytical analysis for comparison
	def analyticalAnalysis(beam_path):
		secondMoment = np.load(beam_path / 'label' / 'secondMoment.npy') # ? (depending on the unit selected in FreeCAD i.e. should be mm^4)
		length = np.load(beam_path / 'numbers' / 'extrude_length.npy') # mm
		Force = np.load(beam_path / 'numbers' / 'FEAloadCase.npy') # N
		volume = np.load(beam_path / 'label' / 'volume.npy')
		youngs_modulus = 7e7 # [N/mm^2] (Aluminium COMSOL E=70GPa=7e10Pa) note: 1Pa = 0.001 kg/(m*s^2)
		density = 2.7e-6 # 2.7×10^(-6) kg/mm^3 (kilograms per millimeter cubed) or 2700[kg/m^3] (Aluminium COMSOL)

		mybeam = analyticalCantileverbeam(secondMoment = secondMoment, force = Force, elasticity = youngs_modulus,
											density = density, beam_length = length, volume = volume)
		print("Analytical: ", mybeam.getEigenfrequencies())

		ef1 = np.load(beam_path / 'label' / 'ef1.npy')[0] # Hz
		ef2 = np.load(beam_path / 'label' / 'ef2.npy')[0] # Hz
		ef3 = np.load(beam_path / 'label' / 'ef3.npy')[0] # Hz
		print("FEA: ", [ef1, ef2, ef3])
	analyticalAnalysis(dataset_path / str(beam_number))
if __name__ == '__main__':
	main()
