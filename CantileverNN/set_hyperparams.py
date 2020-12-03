'''
@Title: set_yperparams.py
@Authoer: Philippe Wyder
@Description: Script to generate hyper-parameter json file
'''

import json

print("Current hyperparameter configuration file name: hyperparam.json")
outfile = input("Please enter a filename for your hyperparameter file or press [enter] to accept the default name: ") 
if not outfile:
	outfile = 'hyperparam.json'
params = {}
params['num_epochs']            = 150
params['batch_size']            = 120
params['lbl_selection']         = 'ef1'
params['learning_rate']         = 0.0001
params['dataset_path']          = '/datasets/TwistedBeamDS'
params['img_name']		= 'img_agrayscale_128.jpg'
params['dataset_split']		= (64,16,20)
params['NeuralNetwork']		= 'ConvNetExtended'
# Prompt user to update file parameters
for key in params.keys()
	print("Parameter '{}' is set to: {}".format(key, params[key])
	if input("Would you like to change '{}'?[y/n]".format(key).lower() =='y'):
		params[key] = type(params[key])(input)

# Non Alterable Parameters
#(These settings are currently only implemented for the following configuration)

params['lr_isadaptive']         = False
params['lr_decreasefactor']     = 0.5 
params['normalization_range']   = (0,1)
params['norm_mode']		= 'pass_through'


with open(outfile, 'w') as fp:
    json.dump(params, fp)

print("Stored Hyper Parameters in ", outfile)
