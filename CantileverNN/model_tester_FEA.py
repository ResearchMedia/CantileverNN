'''
@Title: Model Tester MOI
@Description: Script to validation pre-trained cantilever beam models
@Author: Philippe Wyder (pmw2125@columbia.edu)
'''
from __future__ import print_function, division
import torch 
import torchvision
from torchvision import transforms, utils
from pathlib import Path
from skimage import io, transform
import os
import numpy as np
import matplotlib.pyplot as plt
import csv, datetime
import json
import argparse
from importlib import import_module
from FEABeamDataset import *
from ConvNet import ConvNet

class ModelTester():
  """model tester class"""
  def __init__(self, model_dict, data_loader_dict, num_classes, GPU_id = 0, out_dir = Path('.')):
    self.model_state_dict = model_dict['model_state_dict']
    self.hyper_params = model_dict['hyperparam_dict']
    self.norm_dict = model_dict['norm_dict']
    self.dataloaders = data_loader_dict
    self.GPU = GPU_id
    self.file_root = 'evaluation_' + model_dict['file_name_details']
    self.out_dir = out_dir
    self.lbl_header = model_dict['lbl_header']
    self.num_classes = num_classes
    self.img_res = model_dict['img_res']
    #self.lbl_normalized = True
    if 'NeuralNetwork' in self.hyper_params:
      neuralnetwork = self.hyper_params['NeuralNetwork']
      my_module = import_module(neuralnetwork)
      print("Loaded Neural Network from ", neuralnetwork, ".py file")
      global ConvNet # to overwrite the initially loaded model
      ConvNet = getattr(my_module, "ConvNet")
  def evaluate(self):
    # Device configuration
    gpu_id = 'cuda:' + str(self.GPU)
    device = torch.device(gpu_id if torch.cuda.is_available() else 'cpu')
    print('Running Network on ', device)

    # set the number of classes to estimate
    num_classes = self.num_classes
    print("Number of classes: ", num_classes)

    # Load the model checkpoint
    model = ConvNet(num_classes = num_classes, num_img_layers = 1, img_res = self.img_res).to(device)
    model.to(device)
    model.load_state_dict(self.model_state_dict)

    # Evaluation Summary File
    evaluation_summary_file = self.out_dir / (self.file_root + '_summary' + '.csv')
    evaluation_summary = []

    epsilon = torch.tensor(1e-12, dtype = torch.float)
    # Test the model
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
      for ds in self.dataloaders.keys():
        start_time = datetime.datetime.now()
        current_dl = self.dataloaders[ds]
        log_file = self.out_dir / (self.file_root + '_' + ds + '.csv')
        abs_err = 0
        abs_perr = 0
        tot_nr_lbls = 0
        evaluation_log = {}
        # define field names for CSV outfile
        fieldnames = ['index','volume','AvgPErr', 'AvgErr']
        for i in range(0,num_classes):
          fieldnames.append('lbl ' + str(i) + " " + self.lbl_header[i])
          fieldnames.append('model ' + str(i) + " " + self.lbl_header[i])
          fieldnames.append('perr ' + str(i) + " " + self.lbl_header[i])
        fieldnames[2: ] = sorted(fieldnames[2: ])
        #print(fieldnames)
        with open(log_file, 'w') as csvfile:
          writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
          writer.writeheader()
          # added extrude length but am not using it yet
          for element in current_dl:
            images, labels, extrude_length, volume = element['img'], element['lbl'], element['extrude_length'], element['volume']
            images = images.to(device)

            labels = labels.reshape(labels.shape[0],num_classes)
            outputs = model(images)
            outputs = outputs.to('cpu')

            # denormalize values
#            if self.lbl_normalized:
            labels = denormalize(labels, lower_bound = self.norm_dict['range_min'], upper_bound = self.norm_dict['range_max'],
                       lbl_min = self.norm_dict['min'], lbl_max = self.norm_dict['max'],
                       avg = self.norm_dict['avg'], stdev = self.norm_dict['std'],
                       logmask = self.norm_dict['logmask'], mode = self.norm_dict['mode'])
            outputs = denormalize(outputs, lower_bound = self.norm_dict['range_min'], upper_bound = self.norm_dict['range_max'],
                       lbl_min = self.norm_dict['min'], lbl_max = self.norm_dict['max'],
                       avg = self.norm_dict['avg'], stdev = self.norm_dict['std'], 
                       logmask = self.norm_dict['logmask'], mode = self.norm_dict['mode'])

            tot_nr_lbls += labels.size(0)
            error = abs(outputs - labels) 
            percent_error = np.divide(abs(outputs - labels), torch.max(labels, epsilon)) 
            abs_err += (error.sum(dim = 1)).sum().item()
            abs_perr += (percent_error.sum(dim = 1)).sum().item()

            # If one dimensional prediction write out output for analysis
            for idx in range(labels.size(0)):
              csv_line = {'index': idx}
              for i in range(0,num_classes):
                csv_line['lbl ' + str(i) + " " + self.lbl_header[i]] = labels[idx][i].item()
                csv_line['model ' + str(i) + " " + self.lbl_header[i]] = outputs[idx][i].item()
                csv_line['perr ' + str(i) + " " + self.lbl_header[i]] = percent_error[idx][i].item() 
              csv_line['volume'] = volume[idx].item() 
              csv_line['AvgErr'] = error[idx].sum().item()/num_classes
              csv_line['AvgPErr'] = percent_error[idx].sum().item()/num_classes

              #print(csv_line)
              writer.writerow(csv_line)

        print('{} Error of the model on the {} test images (MAE): {}'.format(ds, tot_nr_lbls,
            abs_err / tot_nr_lbls))
        print('{} Error of the model on the {} test images (MAPE): {} %'.format(ds, tot_nr_lbls,
            100 * abs_perr / tot_nr_lbls))
        evaluation_summary.append(
            { 'Dataset': ds,
              'Set_Size': tot_nr_lbls,
              'MAE':abs_err / tot_nr_lbls,
              'MAPE':100 * abs_perr / tot_nr_lbls,
            }
          )
        # Stop Timing
        end_time = datetime.datetime.now()
        print('Evaluation took (seconds): {}'.format(end_time-start_time))

    print("Writing Summary File")
    with open(evaluation_summary_file, 'w') as summary_file:
      summary_fieldnames = ['Dataset', 'Set_Size', 'MAE', 'MAPE']
      writer = csv.DictWriter(summary_file, fieldnames=summary_fieldnames)
      writer.writeheader()
      writer.writerows(evaluation_summary)

        #implement evaluate function
#  def setLblNormalized(self, is_normalized):
#    if is_normalized:
#      self.lbl_normalized = True
#    else:
#      self.lbl_normalized = False

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='PyTorch Neural Network Parser')
  #parser.add_argument('--hyperparam-json', type=str, default=None, metavar='hyperparam.json',
  #                    help='Provide a .json file with a dictionary of required hyper parameters e.g. hyperparam.json')
  parser.add_argument('--model-name', type=str, default=None, metavar='model.ckpt',
                      help='Provide a .ckpt model file.')
  parser.add_argument('--GPU', type=int, default=0, metavar='Any valid GPU id e.g. 0',
                      help='Provide a GPU id e.g. 0,1,etc.')
  args = parser.parse_args()
  # Load requested model dictionary
  # if model_name provided, load the requested model
  if args.model_name:
    model_name = args.model_name
    model_dict = torch.load(model_name)
    hyper_params = model_dict["hyperparam_dict"]
  else:
    print("No model name argument provided. Try to load: ", model_name)
    quit()

  if hyper_params:
    num_epochs    = hyper_params['num_epochs']
    batch_size    = hyper_params['batch_size']
    lbl_selection = hyper_params['lbl_selection']
    learning_rate = hyper_params['learning_rate']
    lr_isadaptive = hyper_params['lr_isadaptive']
    lr_decreasefactor = hyper_params['lr_decreasefactor']
    normalization_range = hyper_params['normalization_range']
    logmask = hyper_params['logmask']
    norm_dict = {}
    norm_dict['mode'] = hyper_params['norm_mode']
    dataset_path = hyper_params['dataset_path']
    img_name = hyper_params['img_name']
    dataset_split = hyper_params['dataset_split']
  else:
    print("Missing Hyperparameters for evaluation. hyper_params is", hyper_params)
    quit()

  # initalize datasets
  train_dataset = FEABeamDataset(root_dir = dataset_path, 
                                             img_name = img_name,
                                             lbl_selection = lbl_selection, split = dataset_split, set_type = 'train', transform=ToTensor())
  print("Length of train dataset: ", len(train_dataset))
  # set the number of classes to estimate
  num_classes = train_dataset.getNumClasses()
  print("Number of classes: ", num_classes)

  # load and scale validation dataset
  validation_dataset = FEABeamDataset(root_dir = dataset_path, 
                                             img_name = img_name,
                                             lbl_selection = lbl_selection, split = dataset_split, set_type = 'validation', transform=ToTensor())
  print("Length of validation dataset: ", len(validation_dataset))

  # Data loader
  train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                             batch_size=batch_size, 
                                             shuffle=True)

  validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset,
                                            batch_size=batch_size, 
                                            shuffle=False)

  # load and scale test dataset if requested
  if len(dataset_split) == 3:
    test_dataset = FEABeamDataset(root_dir = dataset_path, 
                                               img_name = img_name,
                                               lbl_selection = lbl_selection, split = dataset_split, set_type = 'test', transform=ToTensor())
    print("Length of test dataset: ", len(test_dataset))
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size = batch_size,
                                              shuffle = False)
  # get img resolution
  #img_res = train_dataset[0]['img'].shape[2]
  #print("Image resolution used: ", img_res)
  #model = ConvNet(num_classes=num_classes, num_img_layers = 1, img_res = img_res).to(device)

  # initialized dataloader dict
  dataloaders = {}
  dataloaders['train'] = train_loader
  dataloaders['validation'] = validation_loader
  dataloaders['test'] = test_loader

  mt = ModelTester(model_dict, dataloaders, num_classes, GPU_id = args.GPU)
  # Lbls are not normalized
  mt.setLblNormalized(False)
  mt.evaluate()