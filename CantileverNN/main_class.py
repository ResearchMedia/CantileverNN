from __future__ import print_function, division
import torch 
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage import io, transform
import csv, datetime
from FEABeamDataset import *
from ConvNet import ConvNet
from model_tester_FEA import ModelTester
import json
import argparse
from importlib import import_module
import sys, select, tty, termios # imports for key inputs

class ModelTrainer():
    def __init__(self, hyperparam_dict, GPU_id = 0, model_file = None):
        if not hyperparam_dict:
            return 
        if model_file is not None:
            print("Not Implemented model_file loading")
            return None
            # ToDo: implement propper way of continuing to train a model
            # Uncomment to continue previousily trained model
            #model.load_state_dict(torch.load('model.ckpt')) 
            # load existing model and keep training
        neuralnetwork = hyperparam_dict['NeuralNetwork']
        my_module = import_module(neuralnetwork)
        print("Loaded Neural Network from ", neuralnetwork, ".py file")
        global ConvNet # to overwrite the initially loaded model
        ConvNet = getattr(my_module, "ConvNet")

        #tolerance = 0.01
        self.varstop = 0
        self.norm_dict = {}
        self.hyperparam_dict = hyperparam_dict
        # load hyperparameters
        self.num_epochs    = hyperparam_dict['num_epochs']
        self.batch_size    = hyperparam_dict['batch_size']
        self.lbl_selection = hyperparam_dict['lbl_selection']
        self.learning_rate = hyperparam_dict['learning_rate']
        self.lr_isadaptive = hyperparam_dict['lr_isadaptive']
        self.lr_decreasefactor = hyperparam_dict['lr_decreasefactor']
        self.norm_dict['range_min'] = hyperparam_dict['normalization_range'][0] 
        self.norm_dict['range_max'] = hyperparam_dict['normalization_range'][1]
        self.norm_dict['logmask'] = hyperparam_dict['logmask']
        self.norm_dict['mode'] = hyperparam_dict['norm_mode']
        self.dataset_path = Path(hyperparam_dict['dataset_path'])
        self.img_name = hyperparam_dict['img_name']
        self.dataset_split = hyperparam_dict['dataset_split']

        runtime = str(datetime.datetime.now().strftime("%Y%m%d%H%M"))
        self.out_dir = Path('{}/{}/{}/BS{}.{}'.format(
            self.dataset_path.name, neuralnetwork, self.lbl_selection, self.batch_size, self.learning_rate))
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.file_name_details = self.lbl_selection + self.img_name[:-4] + self.dataset_path.name + runtime 
        self.log_file = self.out_dir / ('log' + self.file_name_details + '.csv')
        self.best_model_file = self.out_dir/ ('best_model_' + self.file_name_details + '.ckpt')
        self.model_file = self.out_dir / ('model_' + self.file_name_details + '.ckpt')

        # Device configuration
        self.gpu_id = GPU_id
        self.device = torch.device('cuda:' + str(self.gpu_id) if torch.cuda.is_available() else 'cpu')
        print('Running Network on ', self.device)
    def loadData(self):
        print("Loading Dataset...")
        # initalize datasets
        self.train_dataset = FEABeamDataset(root_dir = self.dataset_path, img_name = self.img_name, 
                                            lbl_selection = self.lbl_selection, split = self.dataset_split, 
                                            set_type = 'train', transform=ToTensor())
        print("Length of train dataset: ", len(self.train_dataset))
        # set the number of classes to estimate
        self.num_classes = self.train_dataset.getNumClasses()
        print("Number of classes: ", self.num_classes)

        # Scale Dataset
        print("Normalization Mode: ", self.norm_dict['mode'])
        self.norm_dict['min'] = self.train_dataset.getLblAmin()
        self.norm_dict['max'] = self.train_dataset.getLblAmax()
        self.norm_dict['avg'] = self.train_dataset.getLblAvg()
        self.norm_dict['std'] = self.train_dataset.getLblStd()


        self.train_dataset.normalizeLabels(lower_bound = self.norm_dict['range_min'], upper_bound = self.norm_dict['range_max'],
                                     lbl_min = self.norm_dict['min'], lbl_max = self.norm_dict['max'],
                                     avg = self.norm_dict['avg'], stdev = self.norm_dict['std'],
                                     logmask = self.norm_dict['logmask'], mode = self.norm_dict['mode'])
        # load and scale validation dataset
        self.validation_dataset = FEABeamDataset(root_dir = self.dataset_path, img_name = self.img_name,
                                                   lbl_selection = self.lbl_selection, split = self.dataset_split,
                                                   set_type = 'validation', transform=ToTensor())
        print("Length of validation dataset: ", len(self.validation_dataset))
        self.validation_dataset.normalizeLabels(lower_bound = self.norm_dict['range_min'], upper_bound = self.norm_dict['range_max'],
                         lbl_min = self.norm_dict['min'], lbl_max = self.norm_dict['max'],
                         avg = self.norm_dict['avg'], stdev = self.norm_dict['std'],
                         logmask = self.norm_dict['logmask'], mode = self.norm_dict['mode'])

        # Data loader
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                   batch_size=self.batch_size, 
                                                   shuffle=True)

        self.validation_loader = torch.utils.data.DataLoader(dataset=self.validation_dataset,
                                                  batch_size=self.batch_size, 
                                                  shuffle=False)

        # load and scale test dataset if requested
        if len(self.dataset_split) == 3:
            self.test_dataset = FEABeamDataset( root_dir = self.dataset_path, img_name = self.img_name,
                                                lbl_selection = self.lbl_selection, split = self.dataset_split,
                                                set_type = 'test', transform=ToTensor())
            self.train_dataset.normalizeLabels( lower_bound = self.norm_dict['range_min'], upper_bound = self.norm_dict['range_max'],
                                                lbl_min = self.norm_dict['min'], lbl_max = self.norm_dict['max'],
                                                avg = self.norm_dict['avg'], stdev = self.norm_dict['std'],
                                                logmask = self.norm_dict['logmask'], mode = self.norm_dict['mode'])
            print("Length of test dataset: ", len(self.test_dataset))
            self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset,
                                                      batch_size = self.batch_size,
                                                      shuffle = False)
        # get img resolution
        self.img_res = self.train_dataset[0]['img'].shape[2]
        print("Image resolution used: ", self.img_res)
        self.model = ConvNet(num_classes=self.num_classes, num_img_layers = 1, img_res = self.img_res).to(self.device)

    # stopping fraction from arguments
    def setVarStop(self, varstop=0):
        print("set variable stopping condition to: ", varstop)
        self.varstop = varstop
    def setGPU(self, GPU_id = 0):
        print("Set GPU id to : ", GPU_id)
        # Device configuration
        self.gpu_id = GPU_id
        self.device = torch.device('cuda:' + str(self.gpu_id) if torch.cuda.is_available() else 'cpu')
        print('Running Network on ', device)
    def getvalidationloss(self):
        with torch.no_grad():
            tot_validation_loss = 0
            validation_length = len(self.validation_loader)
            # added extrude length but am not using it yet
            for element in self.validation_loader:
                images, labels, extrude_length = element['img'], element['lbl'], element['extrude_length']
                images = images.to(self.device)

                #labels = labels.reshape(labels.shape[0],num_classes)
                labels = labels.to(self.device)
                outputs = self.model(images)
                # reshape labels (necessary! so size is [batch_size, num_classes])
                labels = labels.reshape(labels.shape[0],self.num_classes)
                loss = self.criterion(outputs, labels)
                tot_validation_loss += loss.item()
        return (tot_validation_loss / validation_length)
    def setLoss(self, loss_function = 'MSELoss'):
        print("Initialize Loss Function: ", loss_function)
        if loss_function == 'MSELoss':
            self.criterion = nn.MSELoss()
        elif loss_function == 'SmoothL1Loss':
            self.criterion = nn.SmoothL1Loss()
        else:
            print('Loss function {} not available. No Loss Set!'.format(loss_function))
    def setOptim(self, optimizer = 'Adam'):
        print("Set Optimizer: ", optimizer)
        if optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        else:
            print('Optimizer {} not available. No Optimizer Set!'.format(optimizer))
    def setScheduler(self, scheduler_type = 'static'):
        print("Set Scheduler to ", scheduler_type)
        if scheduler_type == 'static':
            lambda1 = lambda epoch: 1 
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda1)
        elif self.lr_isadaptive:
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, self.lr_decreasefactor, last_epoch=-1)
        else:
            print('Scheduler {} not available. No Scheduler Set!'.format(scheduler_type))

    def trainModel(self):
        print("Start Training Model")
        # start timer
        start_time = datetime.datetime.now()

        # Train the model
        nr_batches = len(self.train_loader) # nr_batches*batch_size = nr_datapoints in train_loader
        header = ['Epoch', 'TotTrainLoss', 'TotValidationLoss', 'LearningRate']
        self.log = []
        best_tot_validation_loss = float("inf")
        self.best_model = {}


        # user input handling
        def isData():
            return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])
        old_settings = termios.tcgetattr(sys.stdin) # save old handler
        try:
            tty.setcbreak(sys.stdin.fileno())

            ## initialize train losses
            #prev_tot_train_loss = 0
            self.tot_train_loss = 0
            self.epoch = 0
            while (True):

                # user input handling
                if isData():
                    c = sys.stdin.read(1)
                    print(c)
                    if c == '\x07':         # x1b is ESC
                        print("[CTRL+G] Training abborted. Saving Files...")
                        break
                
                # added extrude length but am not using it yet
                #prev_tot_train_loss = tot_train_loss
                self.tot_train_loss = 0
                for i, element in enumerate(self.train_loader):
                    images, labels = element['img'], element['lbl']

                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    
                    # Forward pass
                    outputs = self.model(images)
                    # reshape labels !necessary for single class models! so [120] becomes [120,1]
                    labels = labels.reshape(labels.shape[0], self.num_classes)
                    #print("Label vs. Output Example: lbl1:{} lbl2:{} out1:{} out2:{}".format(labels[0,0], labels[0,1], outputs[0,0], outputs[0,1]))
                    loss = self.criterion(outputs, labels)

                    self.tot_train_loss += loss.item()
                    # Backward and optimize
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                self.tot_train_loss = self.tot_train_loss/nr_batches
                self.tot_validation_loss = self.getvalidationloss()
                # reduce learning rate dynamically
                self.scheduler.step()
                print ('Epoch [{}/{}], \tStep [{}/{}], \tTrain Loss: {:.4e}, \tValidation Loss: {:.4e}, \tLearning Rate: {:.32f}' 
                       .format(self.epoch+1, self.num_epochs, nr_batches, nr_batches, self.tot_train_loss, self.tot_validation_loss, self.scheduler.get_lr()[0]))
                #print("Learning Rate In List Form: ", scheduler.get_lr())
                self.log.append([self.epoch, self.tot_train_loss, self.tot_validation_loss, self.scheduler.get_lr()[0]])
                # store best models
                if self.epoch > 0:
                    if self.best_tot_validation_loss > self.tot_validation_loss:
                        self.best_tot_validation_loss = self.tot_validation_loss
                        self.best_model = self.getCurrentModelDict() 
                        print("Best model: ", self.best_tot_validation_loss, "\tepoch: ", self.epoch)
                        if self.epoch%10 == 0:
                            torch.save(self.best_model, self.best_model_file)
                else:
                    self.best_tot_validation_loss = self.tot_validation_loss
                    self.best_model = self.getCurrentModelDict()
                    print("Best model: ", self.best_tot_validation_loss, "\tepoch: ", self.epoch)

                # variable stopping condition
                if self.varstop < 1.0 and self.varstop > 1e-12:
                    if self.epoch < self.num_epochs:
                        # stop the model if it hasn't improved within epoch*varstop
                        if self.best_model['epoch'] < self.epoch*self.varstop and self.epoch > self.num_epochs*self.varstop:
                            print("Model has not improved since {} epochs".format(self.epoch-self.best_model['epoch']))
                            print("Stopping training process")
                            break;
                        if not self.epoch + 1 < self.num_epochs:
                            self.num_epochs *= 2
                    else:
                        break
                elif self.varstop > 1.0:
                    if self.num_epochs < self.epoch or self.varstop < (self.epoch - self.best_model['epoch']):
                        print("Model has not improved since {} epochs".format(self.epoch-self.best_model['epoch']))
                        print("Stopping training process")
                        break
                    else:
                        if not self.epoch + 1 < self.num_epochs:
                            self.num_epochs *= 2

                else:
                    if self.num_epochs < self.epoch:
                        print("Model has not improved since {} epochs".format(self.epoch-self.best_model['epoch']))
                        print("Stopping training process")
                        break
                self.epoch += 1; # increment epoch



        finally: # restore previous handler
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

        # save best model 
        torch.save(self.best_model, self.best_model_file)


        # Stop Timing
        end_time = datetime.datetime.now()
        print('Training took (Time): {}'.format(end_time-start_time))

        # write log to file
        with open(self.log_file, 'wt') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(header) # write header
            csv_writer.writerows(self.log)
    def evaluateModel(self):
        # initialized dataloader dict
        dataloaders = {}
        dataloaders['train'] = self.train_loader
        dataloaders['validation'] = self.validation_loader
        dataloaders['test'] = self.test_loader

        # Evaluate Best Model
        mt = ModelTester(self.best_model, dataloaders, self.num_classes, GPU_id = self.gpu_id, out_dir = self.out_dir)
        mt.evaluate()
    def getCurrentModelDict(self):
        # Save the model checkpoint
        model_checkpoint = {    'epoch': self.epoch,
                                'model_state_dict': self.model.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'hyperparam_dict': self.hyperparam_dict,
                                'norm_dict': self.norm_dict,
                                'lbl_header': self.train_dataset.getLblHeader(),
                                'file_name_details': self.file_name_details,
                                'img_res': self.img_res,
                                'tot_train_loss': self.tot_train_loss,
                                'tot_validation_loss': self.tot_validation_loss }
        return model_checkpoint

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='PyTorch Neural Network Parser')
    parser.add_argument('--hyperparam-json', type=str, default='hyperparam.json', metavar='hyperparam.json',
                        help='Provide a .json file with a dictionary of required hyper parameters e.g. hyperparam.json')
    parser.add_argument('--GPU', type=int, default=0, metavar='Any valid GPU id e.g. 0',
                        help='Provide a GPU id e.g. 0,1,etc.')
    parser.add_argument('--varstop', type=float, default=0, metavar='1 for variable stopping condition',
                        help='1 to stop variable, 0 to stop after num_epochs')
    parser.add_argument('--model', type=str, default=None, metavar='provide existing model to continue training',
                        help='model_name.ckpt')
    args = parser.parse_args()

    hyper_params = None
    # if a hyper_parameter file has been selected, load it
    if args.hyperparam_json:
        with open(args.hyperparam_json, 'r') as hyper_param_file:
            hyper_params  = json.load(hyper_param_file)
    else:
        if args.model:
            if input("No Hyperparmeter-File provided. Continue?: [y/n]") != 'y':
                print("program terminated")
                quit()
        else:
            print("Neither hyperparameter file nor model provided! Program terminated!")
            quit()

    # get stopping fraction from ARGS
    stopping_fraction = args.varstop
    mytrainer = ModelTrainer(hyperparam_dict = hyper_params, GPU_id = args.GPU, model_file = args.model)
    mytrainer.loadData()
    mytrainer.setVarStop(args.varstop)
    mytrainer.setLoss()
    mytrainer.setOptim()
    mytrainer.setScheduler()
    mytrainer.trainModel()
    mytrainer.evaluateModel()
