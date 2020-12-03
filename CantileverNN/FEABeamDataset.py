'''
Cantilever Beam Dataset Data Loader
@Author: Philippe Wyder (PMW2125@columbia.edu)
@description: dataset of random cantilever beams
'''
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

MAX_VERT = 30 # maximum number of vertices in dataset

class FEABeamDataset(Dataset):
    """Cantilever Beam Dataset"""
    def __init__(self, root_dir, img_name="img_bw.jpg", lbl_selection='DispField_all', split=(80,20), set_type = 'train', transform=None):
        """
        Args:
            root_dir (Path): path to dataset directory
            img_name (string): name of the image used for the dataset
            transform (callable, options): Optional transform to be applied to sample
            labels (list, double): list of folder names (assumes 0 indexing)
        """
        self.root_dir = Path(root_dir)
        self.lbl_header = []
        self.img_name = img_name
        self.lbl_selection = lbl_selection
        self.transform = transform
        self.set_type = set_type
        # list all data point folders and verify sequence completeness
        datafolders = []
        with os.scandir(self.root_dir) as it:
            for entry in it:
                if entry.is_dir():
                    datafolders.append(int(entry.name))
        #datafolders = sorted(list(map(int, os.listdir(self.root_dir))))
        self.datafolders = sorted(datafolders)
        if not (self.datafolders[max(self.datafolders)] == max(self.datafolders)
            and self.datafolders[0] == 0):
            print("datafolder sequence incomplete")
            print("idx = 0: ", self.datafolders[0])
            print("idx = last: ", self.datafolders[max(self.datafolders)])
            return
        self.num_classes = 1
        # Dataset Separation
        self.split = np.array(split)/100
        if not sum(self.split):
            print("Dataset split doesn't add up to 1: ", sum(self.split))
            return
        self.set_type = set_type
        self._split_data()
        # load all labels and numbers
        self.labels = []
        self.is_normalized = False
        self.verts = []
        self.extrude_length = []
        self.load_case = []
        self.volume = []

        for idx in self.folder_range:
            # Load numbers
            numbers_path = (self.root_dir / str(idx) / "numbers")

            # verts are being padded with zeroes to be fixed MAX_VERT size
            padded_verts = np.zeros((MAX_VERT, 2))
            raw_verts = np.load(numbers_path / "verts.npy")
            padded_verts[:raw_verts.shape[0], :raw_verts.shape[1]] = raw_verts
            self.verts.append(padded_verts)

            self.extrude_length.append(np.load(numbers_path / "extrude_length.npy"))
            self.load_case.append(np.load(numbers_path / "FEAloadCase.npy"))
            self.volume.append(np.load(self.root_dir / str(idx) / "label" / "volume.npy"))
            self._getlabel(idx)


        # Convert Labels to numpy array
        self.labels = np.array(self.labels, np.float32)
        self.labels = torch.from_numpy(self.labels)
        # Note: images will not be loaded into memory but read from disk as needed

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # if this is the test set, then offset the index to read the test data
        lbl = self.labels[idx]
        volume = self.volume[idx]
        extrude_length = self.extrude_length[idx]
        verts = self.verts[idx]
        load_case = self.load_case[idx]
        # update index for folder index if test data set
        if self.set_type == 'validation':
            idx = idx + self.sep_idx # offset data to test_data folders
        elif self.set_type == 'test':
            idx = idx + self.sep_idx_2 # offset data to test_data folders
            #print("Test Index: ", idx)
        img = io.imread(self.root_dir / str(idx) / "img" / self.img_name, as_gray=True)
        sample = {  'img': img, 'lbl': lbl,
                    'extrude_length':extrude_length, 'volume':volume,
                    'verts': verts, 'load_case':load_case }
        if self.transform:
            sample = self.transform(sample)
        return sample
    def _getlabel(self, idx):
        # Load Labels (consider placing in separate function)
        if (self.lbl_selection[0:3] == 'MOI'):
            moi = np.load(self.root_dir / str(idx) / "label" / "MatrixOfInertia.npy")
            '''
                Moment of Inertia matrix Labels
            '''
            if (self.lbl_selection == 'MOI_all'):
                self.lbl_header = ['Ixx','Ixy', 'Ixz', 'Iyy', 'Iyz', 'Izz']
                self.labels.append(np.array([moi[0,0], moi[0,1], moi[0,2],
                                                       moi[1,1], moi[1,2],
                                                                 moi[2,2] ])
                               )
                self.num_classes = 6
            elif (self.lbl_selection == 'MOI_Ixx'):
                self.lbl_header = ['Ixx']
                self.labels.append(np.array(moi[0,0]))
            elif (self.lbl_selection == 'MOI_Iyy'):
                self.lbl_header = ['Iyy']
                self.labels.append(np.array(moi[1,1]))
            elif (self.lbl_selection == 'MOI_Izz'):
                self.lbl_header = ['Izz']
                self.labels.append(np.array(moi[2,2]))
            elif (self.lbl_selection == 'MOI_Ixy'):
                self.lbl_header = ['Ixy']
                self.labels.append(np.array(moi[0,1]))
            elif (self.lbl_selection == 'MOI_Ixz'):
                self.lbl_header = ['Ixz']
                self.labels.append(np.array(moi[0,2]))
            elif (self.lbl_selection == 'MOI_Iyz'):
                self.lbl_header = ['Iyz']
                self.labels.append(np.array(moi[1,2]))
            elif (lblselection == 'MOI_IxxIyyVol'):
                totDisp = np.load(self.root_dir / str(idx) / "label" / "totalDisplacementmm.npy")
                totVonMises = np.load(self.root_dir / str(idx) / "label" / "totalVonMises.npy")
                moi = np.load(self.root_dir / str(idx) / "label" / "MatrixOfInertia.npy")
                volume = np.load(self.root_dir / str(idx) / "label" / "volume.npy")
                self.lbl_header = ['Ixx', 'Iyy', 'Izz']
                self.labels.append(np.array([moi[0,0], moi[1,1], volume]))
                self.num_classes = 3
        elif (self.lbl_selection == 'DispField_all'): #Displacement Field [mm] Labels (Volume Maximum) 
            self.lbl_header = ['Tot. X-Disp [mm]', 'Tot. Y-Disp [mm]', 'Tot. Z-Disp [mm]']
            self.labels.append(np.load(self.root_dir / str(idx) / "label" / "displacementFieldmm.npy"))
            self.num_classes = 3
        elif (self.lbl_selection == 'DispField_X'):
            self.lbl_header = ['Tot. X-Disp [mm]']
            self.labels.append(np.load(self.root_dir / str(idx) / "label" / "displacementFieldmm.npy")[0])
        elif (self.lbl_selection == 'DispField_Y'):
            self.lbl_header = ['Tot. Y-Disp [mm]']
            self.labels.append(np.load(self.root_dir / str(idx) / "label" / "displacementFieldmm.npy")[1])
        elif (self.lbl_selection == 'DispField_Z'):
            self.lbl_header = ['Tot. Z-Disp [mm]']
            self.labels.append(np.load(self.root_dir / str(idx) / "label" / "displacementFieldmm.npy")[2])
        elif (self.lbl_selection == 'PrincStrain_all'): #Principal Strain Labels [unit-less] (Volume Maximum) 
            self.lbl_header = ['Princ. Strain X', 'Princ. Strain Y', 'Princ. Strain Z']
            self.labels.append(np.load(self.root_dir / str(idx) / "label" / "principalStrain.npy"))
            self.num_classes = 3
        elif (self.lbl_selection == 'PrincStrain_X'):
            self.lbl_header = ['Princ. Strain X']
            self.labels.append(np.load(self.root_dir / str(idx) / "label" / "principalStrain.npy")[0])
        elif (self.lbl_selection == 'PrincStrain_Y'):
            self.lbl_header = ['Princ. Strain Y']
            self.labels.append(np.load(self.root_dir / str(idx) / "label" / "principalStrain.npy")[1])
        elif (self.lbl_selection == 'Princtrain_Z'):
            self.lbl_header = ['Princ. Strain Z']
            self.labels.append(np.load(self.root_dir / str(idx) / "label" / "principalStrain.npy")[2])
        elif (self.lbl_selection == 'PrincStrain_X_scaled'):
            self.lbl_header = ['Princ. Strain X (scaled)']
            principalStrain = np.load(self.root_dir / str(idx) / "label" / "principalStrain.npy")[0] 
            self.labels.append(principalStrain.astype(float)*1e6)
        elif (self.lbl_selection == 'PrincStrain_Y_scaled'):
            self.lbl_header = ['Princ. Strain Y (scaled)']
            principalStrain = np.load(self.root_dir / str(idx) / "label" / "principalStrain.npy")[1] 
            self.labels.append(principalStrain.astype(float)*1e6)
        elif (self.lbl_selection == 'Princtrain_Z_scaled'):
            self.lbl_header = ['Princ. Strain Z (scaled)']
            principalStrain = np.load(self.root_dir / str(idx) / "label" / "principalStrain.npy")[2] 
            self.labels.append(principalStrain.astype(float)*1e6)
        elif (self.lbl_selection == 'CurlDisp_all'):#Curl Displacement Labels [unit-less] (Volume Maximum) 
            self.lbl_header = ['Curl Disp. X', 'Curl Disp. Y', 'Curl Disp. Z']
            self.labels.append(np.load(self.root_dir / str(idx) / "label" / "curlDisplacement.npy"))
            self.num_classes = 3
        elif (self.lbl_selection == 'CurlDisp_X'):
            self.lbl_header = ['Curl Disp. X']
            self.labels.append(np.load(self.root_dir / str(idx) / "label" / "curlDisplacement.npy")[0])
        elif (self.lbl_selection == 'CurlDisp_Y'):
            self.lbl_header = ['Curl Disp. Y']
            self.labels.append(np.load(self.root_dir / str(idx) / "label" / "curlDisplacement.npy")[1])
        elif (self.lbl_selection == 'CurlDisp_Z'):
            self.lbl_header = ['Curl Disp. Z']
            self.labels.append(np.load(self.root_dir / str(idx) / "label" / "curlDisplacement.npy")[2])
        elif (self.lbl_selection == 'CurlDisp_X_scaled'):
            self.lbl_header = ['Curl Disp. X (scaled)']
            curlDisp = np.load(self.root_dir / str(idx) / "label" / "curlDisplacement.npy")[0]
            self.labels.append(curlDisp.astype(float)*1e3)
        elif (self.lbl_selection == 'CurlDisp_Y_scaled'):
            self.lbl_header = ['Curl Disp. Y (scaled)']
            curlDisp = np.load(self.root_dir / str(idx) / "label" / "curlDisplacement.npy")[1]
            self.labels.append(curlDisp.astype(float)*1e3)
        elif (self.lbl_selection == 'CurlDisp_Z_scaled'):
            self.lbl_header = ['Curl Disp. Z (scaled)']
            curlDisp = np.load(self.root_dir / str(idx) / "label" / "curlDisplacement.npy")[2]
            self.labels.append(curlDisp.astype(float)*1e3)
        elif (self.lbl_selection == 'TotDisp'):#Total Displacement [mm] Label (Volume Maximum) 
            self.lbl_header = ['Tot. Disp. [mm]']
            self.labels.append(np.load(self.root_dir / str(idx) / "label" / "totalDisplacementmm.npy"))
        elif (self.lbl_selection == 'TotVonMises'):#Total Displacement [mm] Label (Volume Maximum) 
            self.lbl_header = ['VM Stress']
            self.labels.append(np.load(self.root_dir / str(idx) / "label" / "totalVonMises.npy"))
        elif (self.lbl_selection == 'TotDispAndStress'):#Total Displacement [mm] Label (Volume Maximum) 
            totDisp = np.load(self.root_dir / str(idx) / "label" / "totalDisplacementmm.npy")
            totVonMises = np.load(self.root_dir / str(idx) / "label" / "totalVonMises.npy")
            self.lbl_header = ['Tot. Disp. [mm]', 'VM Stress']
            self.labels.append(np.array([totDisp, totVonMises], dtype=np.dtype(float)))
            self.num_classes = 2
        elif (self.lbl_selection == 'Volume'):#Total Displacement [mm] Label (Volume Maximum) 
            volume = np.load(self.root_dir / str(idx) / "label" / "volume.npy")
            self.lbl_header = ['Volume [mm^3]']
            self.labels.append(np.array([volume]))
            self.num_classes = 1
        elif (self.lbl_selection == 'DispVol'):#Total Displacement [mm] Label (Volume Maximum) 
            totDisp = np.load(self.root_dir / str(idx) / "label" / "totalDisplacementmm.npy")
            volume = np.load(self.root_dir / str(idx) / "label" / "volume.npy")
            self.lbl_header = ['Tot. Disp. [mm]', 'Volume']
            self.labels.append(np.array([totDisp, volume]))
            self.num_classes = 2
        elif (self.lbl_selection == 'StressVol'):#Total Displacement [mm] Label (Volume Maximum) 
            totVonMises = np.load(self.root_dir / str(idx) / "label" / "totalVonMises.npy")
            volume = np.load(self.root_dir / str(idx) / "label" / "volume.npy")
            self.lbl_header = ['VM Stress', 'Volume']
            self.labels.append(np.array([totVonMises, volume]))
            self.num_classes = 2
        elif (self.lbl_selection == 'DispStressVol'):#Total Displacement [mm] Label (Volume Maximum) 
            totDisp = np.load(self.root_dir / str(idx) / "label" / "totalDisplacementmm.npy")
            totVonMises = np.load(self.root_dir / str(idx) / "label" / "totalVonMises.npy")
            volume = np.load(self.root_dir / str(idx) / "label" / "volume.npy")
            self.lbl_header = ['Tot. Disp. [mm]','VM Stress', 'Volume']
            self.labels.append(np.array([totDisp, totVonMises, volume]))
            self.num_classes = 3
        elif (self.lbl_selection == 'DispStressIxxIyy'):#Total Displacement [mm] Label (Volume Maximum) 
            totDisp = np.load(self.root_dir / str(idx) / "label" / "totalDisplacementmm.npy")
            totVonMises = np.load(self.root_dir / str(idx) / "label" / "totalVonMises.npy")
            moi = np.load(self.root_dir / str(idx) / "label" / "MatrixOfInertia.npy")
            self.lbl_header = ['Tot. Disp. [mm]','VM Stress', 'Ixx', 'Iyy']
            self.labels.append(np.array([totDisp, totVonMises, moi[0,0], moi[1,1]]))
            self.num_classes = 4
        elif (self.lbl_selection == 'DispStressVolIxxIyy'):#Total Displacement [mm] Label (Volume Maximum) 
            totDisp = np.load(self.root_dir / str(idx) / "label" / "totalDisplacementmm.npy")
            totVonMises = np.load(self.root_dir / str(idx) / "label" / "totalVonMises.npy")
            moi = np.load(self.root_dir / str(idx) / "label" / "MatrixOfInertia.npy")
            volume = np.load(self.root_dir / str(idx) / "label" / "volume.npy")
            self.lbl_header = ['Tot. Disp. [mm]','VM Stress', 'Volume', 'Ixx', 'Iyy']
            self.labels.append(np.array([totDisp, totVonMises, volume, moi[0,0], moi[1,1]]))
            self.num_classes = 5
        # eigenfrequency labels
        elif (self.lbl_selection == 'ef1'):
            ef1 = np.load(self.root_dir / str(idx) / "label" / "ef1.npy")
            self.lbl_header = ['lambda1']
            self.labels.append(np.array(ef1[0]))
            self.num_classes = 1
        elif (self.lbl_selection == 'ef2'):
            ef2 = np.load(self.root_dir / str(idx) / "label" / "ef2.npy")
            self.lbl_header = ['lambda2']
            self.labels.append(np.array(ef2[0]))
            self.num_classes = 1
        elif (self.lbl_selection == 'ef3'):
            ef3 = np.load(self.root_dir / str(idx) / "label" / "ef3.npy")
            self.lbl_header = ['lambda3']
            self.labels.append(np.array(ef3[0]))
            self.num_classes = 1
        elif (self.lbl_selection == 'ef12'):#Total Displacement [mm] Label (Volume Maximum) 
            ef1 = np.load(self.root_dir / str(idx) / "label" / "ef1.npy")
            ef2 = np.load(self.root_dir / str(idx) / "label" / "ef2.npy")
            self.lbl_header = ['lambda1', 'lambda2']
            self.labels.append(np.array([ef1[0], ef2[0]]))
            self.num_classes = 2
        elif (self.lbl_selection == 'ef123'):#Total Displacement [mm] Label (Volume Maximum) 
            ef1 = np.load(self.root_dir / str(idx) / "label" / "ef1.npy")
            ef2 = np.load(self.root_dir / str(idx) / "label" / "ef2.npy")
            ef3 = np.load(self.root_dir / str(idx) / "label" / "ef3.npy")
            self.lbl_header = ['lambda1', 'lambda2', 'lambda3']
            self.labels.append(np.array([ef1[0], ef2[0], ef3[0]]))
            self.num_classes = 3
        elif (self.lbl_selection == 'ef1_lambda_omega'):# EF1 & Angular Frequency Label (Volume Maximum) 
            ef1 = np.load(self.root_dir / str(idx) / "label" / "ef1.npy")
            self.lbl_header = ['lambda1', 'omega1']
            self.labels.append(np.array([ef1[0], ef1[2]]))
            self.num_classes = 2
        elif (self.lbl_selection == 'ef12_lambda_omega'):# EF12 & Angular Frequency Label (Volume Maximum) 
            ef1 = np.load(self.root_dir / str(idx) / "label" / "ef1.npy")
            ef2 = np.load(self.root_dir / str(idx) / "label" / "ef2.npy")
            self.lbl_header = ['lambda1', 'omega1', 'lambda2', 'omega2']
            self.labels.append(np.array([ef1[0], ef1[2], ef2[0], ef2[2]]))
            self.num_classes = 4
        elif (self.lbl_selection == 'ef123_lambda_omega'):# EF123 & Angular Frequency Label (Volume Maximum) 
            ef1 = np.load(self.root_dir / str(idx) / "label" / "ef1.npy")
            ef2 = np.load(self.root_dir / str(idx) / "label" / "ef2.npy")
            ef3 = np.load(self.root_dir / str(idx) / "label" / "ef3.npy")
            self.lbl_header = ['lambda1', 'omega1', 'lambda2', 'omega2', 'lambda3', 'omega3']
            self.labels.append(np.array([ef1[0], ef1[2], ef2[0], ef2[2], ef3[0], ef3[2]]))
            self.num_classes = 6
        elif (self.lbl_selection == 'DispStressEf1'):#Total Displacement [mm] Label (Volume Maximum) 
            totDisp = np.load(self.root_dir / str(idx) / "label" / "totalDisplacementmm.npy")
            totVonMises = np.load(self.root_dir / str(idx) / "label" / "totalVonMises.npy")
            ef1 = np.load(self.root_dir / str(idx) / "label" / "ef1.npy")
            self.lbl_header = ['Tot. Disp. [mm]','VM Stress', 'lambda1']
            self.labels.append(np.array([totDisp, totVonMises, ef1[0]]))
            self.num_classes = 3
        elif (self.lbl_selection == 'DispStressEf1Vol'):#Total Displacement [mm] Label (Volume Maximum) 
            totDisp = np.load(self.root_dir / str(idx) / "label" / "totalDisplacementmm.npy")
            totVonMises = np.load(self.root_dir / str(idx) / "label" / "totalVonMises.npy")
            ef1 = np.load(self.root_dir / str(idx) / "label" / "ef1.npy")
            volume = np.load(self.root_dir / str(idx) / "label" / "volume.npy")
            self.lbl_header = ['Tot. Disp. [mm]','VM Stress', 'lambda1', 'Volume']
            self.labels.append(np.array([totDisp, totVonMises, ef1[0], volume]))
            self.num_classes = 4
        elif (self.lbl_selection == 'ef1_empf'):
            ef1_empf = np.load(self.root_dir / str(idx) / "label" / "empf1.npy")
            self.lbl_header = [ 'lambda1',
                                'EMM1 X-transl. [kg]', 
                                'EMM1 Y-transl. [kg]',
                                'EMM1 Z-transl. [kg]',
                                'EMM1 X-rot. [kg*m^2]',
                                'EMM1 Y-rot. [kg*m^2]',
                                'EMM1 Z-rot. [kg*m^2]']
            self.labels.append(ef1_empf)
            self.num_classes = 7
        elif (self.lbl_selection == 'ef2_empf'):
            ef2_empf = np.load(self.root_dir / str(idx) / "label" / "empf2.npy")
            self.lbl_header = [ 'lambda2',
                                'EMM2 X-transl. [kg]', 
                                'EMM2 Y-transl. [kg]',
                                'EMM2 Z-transl. [kg]',
                                'EMM2 X-rot. [kg*m^2]',
                                'EMM2 Y-rot. [kg*m^2]',
                                'EMM2 Z-rot. [kg*m^2]']
            self.labels.append(ef2_empf)
            self.num_classes = 7
        elif (self.lbl_selection == 'ef3_empf'):
            ef3_empf = np.load(self.root_dir / str(idx) / "label" / "empf3.npy")
            self.lbl_header = [ 'lambda3',
                                'EMM3 X-transl. [kg]', 
                                'EMM3 Y-transl. [kg]',
                                'EMM3 Z-transl. [kg]',
                                'EMM3 X-rot. [kg*m^2]',
                                'EMM3 Y-rot. [kg*m^2]',
                                'EMM3 Z-rot. [kg*m^2]']
            self.labels.append(ef3_empf)
            self.num_classes = 7
        elif (self.lbl_selection == 'ef123_empf'):
            ef1_empf = np.load(self.root_dir / str(idx) / "label" / "empf1.npy")
            ef2_empf = np.load(self.root_dir / str(idx) / "label" / "empf2.npy")
            ef3_empf = np.load(self.root_dir / str(idx) / "label" / "empf3.npy")
            self.lbl_header = [ 'lambda1',
                                'EMM1 X-transl. [kg]', 
                                'EMM1 Y-transl. [kg]',
                                'EMM1 Z-transl. [kg]',
                                'EMM1 X-rot. [kg*m^2]',
                                'EMM1 Y-rot. [kg*m^2]',
                                'EMM1 Z-rot. [kg*m^2]',
                                'lambda2',
                                'EMM2 X-transl. [kg]', 
                                'EMM2 Y-transl. [kg]',
                                'EMM2 Z-transl. [kg]',
                                'EMM2 X-rot. [kg*m^2]',
                                'EMM2 Y-rot. [kg*m^2]',
                                'EMM2 Z-rot. [kg*m^2]',
                                'lambda3',
                                'EMM3 X-transl. [kg]', 
                                'EMM3 Y-transl. [kg]',
                                'EMM3 Z-transl. [kg]',
                                'EMM3 X-rot. [kg*m^2]',
                                'EMM3 Y-rot. [kg*m^2]',
                                'EMM3 Z-rot. [kg*m^2]']
            self.labels.append(np.concatenate((ef1_empf, ef2_empf, ef3_empf)))
            self.num_classes = 7*3
        elif (self.lbl_selection == 'ef1_npf'):
            ef1_npf = np.load(self.root_dir / str(idx) / "label" / "npf1.npy")
            self.lbl_header = [ 'lambda1',
                                'NPF1 X-transl. [1]', 
                                'NPF1 Y-transl. [1]',
                                'NPF1 Z-transl. [1]',
                                'NPF1 X-rot. [1]',
                                'NPF1 Y-rot. [1]',
                                'NPF1 Z-rot. [1]']
            self.labels.append(ef1_npf)
            self.num_classes = 7
        elif (self.lbl_selection == 'ef2_npf'):
            ef2_npf = np.load(self.root_dir / str(idx) / "label" / "npf2.npy")
            self.lbl_header = [ 'lambda2',
                                'NPF2 X-transl. [1]', 
                                'NPF2 Y-transl. [1]',
                                'NPF2 Z-transl. [1]',
                                'NPF2 X-rot. [1]',
                                'NPF2 Y-rot. [1]',
                                'NPF2 Z-rot. [1]']
            self.labels.append(ef2_npf)
            self.num_classes = 7
        elif (self.lbl_selection == 'ef3_npf'):
            ef3_npf = np.load(self.root_dir / str(idx) / "label" / "npf3.npy")
            self.lbl_header = [ 'lambda3',
                                'NPF3 X-transl. [1]', 
                                'NPF3 Y-transl. [1]',
                                'NPF3 Z-transl. [1]',
                                'NPF3 X-rot. [1]',
                                'NPF3 Y-rot. [1]',
                                'NPF3 Z-rot. [1]']
            self.labels.append(ef3_npf)
            self.num_classes = 7
        elif (self.lbl_selection == 'ef123_npf'):
            ef1_npf = np.load(self.root_dir / str(idx) / "label" / "npf1.npy")
            ef2_npf = np.load(self.root_dir / str(idx) / "label" / "npf2.npy")
            ef3_npf = np.load(self.root_dir / str(idx) / "label" / "npf3.npy")
            self.lbl_header = [ 'lambda1',
                                'NPF1 X-transl. [1]', 
                                'NPF1 Y-transl. [1]',
                                'NPF1 Z-transl. [1]',
                                'NPF1 X-rot. [1]',
                                'NPF1 Y-rot. [1]',
                                'NPF1 Z-rot. [1]',
                                'lambda2',
                                'NPF2 X-transl. [1]', 
                                'NPF2 Y-transl. [1]',
                                'NPF2 Z-transl. [1]',
                                'NPF2 X-rot. [1]',
                                'NPF2 Y-rot. [1]',
                                'NPF2 Z-rot. [1]',
                                'lambda3',
                                'NPF3 X-transl. [1]', 
                                'NPF3 Y-transl. [1]',
                                'NPF3 Z-transl. [1]',
                                'NPF3 X-rot. [1]',
                                'NPF3 Y-rot. [1]',
                                'NPF3 Z-rot. [1]']
            self.labels.append(np.concatenate((ef1_npf, ef2_npf, ef3_npf)))
            self.num_classes = 7*3
        elif (self.lbl_selection == 'npf1'):
            npf1 = np.load(self.root_dir / str(idx) / "label" / "npf1.npy")
            self.lbl_header = [ 'NPF1 X-transl. [1]', 
                                'NPF1 Y-transl. [1]',
                                'NPF1 Z-transl. [1]',
                                'NPF1 X-rot. [1]',
                                'NPF1 Y-rot. [1]',
                                'NPF1 Z-rot. [1]']
            self.labels.append(npf1[1:])
            self.num_classes = 6
        elif (self.lbl_selection == 'npf1_TRMS'):
            npf1 = np.load(self.root_dir / str(idx) / "label" / "npf1.npy")
            self.lbl_header = ['npf1_TRMS']
            npf1_squared = npf1.astype(float)[1:4]**2
            self.labels.append(np.sqrt(np.mean(npf1_squared)))
            self.num_classes = 1
        elif (self.lbl_selection == 'npf1_RRMS'):
            npf1 = np.load(self.root_dir / str(idx) / "label" / "npf1.npy")
            self.lbl_header = ['npf1_RRMS']
            npf1_squared = npf1.astype(float)[4:7]**2
            self.labels.append(np.sqrt(np.mean(npf1_squared)))
            self.num_classes = 1
        elif (self.lbl_selection == 'npf1_RMS'):
            npf1 = np.load(self.root_dir / str(idx) / "label" / "npf1.npy")
            self.lbl_header = ['npf1_RMS']
            npf1_squared = npf1.astype(float)[1:7]**2
            self.labels.append(np.sqrt(np.mean(npf1_squared)))
            self.num_classes = 1
        elif (self.lbl_selection == 'npf2_TRMS'):
            npf2 = np.load(self.root_dir / str(idx) / "label" / "npf2.npy")
            self.lbl_header = ['npf2_TRMS']
            npf2_squared = npf2.astype(float)[1:4]**2
            self.labels.append(np.sqrt(np.mean(npf2_squared)))
            self.num_classes = 1
        elif (self.lbl_selection == 'npf2_RRMS'):
            npf2 = np.load(self.root_dir / str(idx) / "label" / "npf2.npy")
            self.lbl_header = ['npf2_RRMS']
            npf2_squared = npf2.astype(float)[4:7]**2
            self.labels.append(np.sqrt(np.mean(npf2_squared)))
            self.num_classes = 1
        elif (self.lbl_selection == 'npf2_RMS'):
            npf2 = np.load(self.root_dir / str(idx) / "label" / "npf2.npy").astype(float) 
            self.lbl_header = ['npf2_RMS']
            self.labels.append(np.sqrt(np.mean(npf2[1:7]**2)))
            self.num_classes = 1
        elif (self.lbl_selection == 'npf3_TRMS'):
            npf3 = np.load(self.root_dir / str(idx) / "label" / "npf3.npy")
            self.lbl_header = ['npf3_TRMS']
            npf3_squared = npf3.astype(float)[1:4]**2
            self.labels.append(np.sqrt(np.mean(npf3_squared)))
            self.num_classes = 1
        elif (self.lbl_selection == 'npf3_RRMS'):
            npf3 = np.load(self.root_dir / str(idx) / "label" / "npf3.npy")
            self.lbl_header = ['npf3_RRMS']
            npf3_squared = npf3.astype(float)[4:7]**2
            self.labels.append(np.sqrt(np.mean(npf3_squared)))
            self.num_classes = 1
        elif (self.lbl_selection == 'npf3_RMS'):
            npf3 = np.load(self.root_dir / str(idx) / "label" / "npf3.npy").astype(float) 
            self.lbl_header = ['npf3_RMS']
            self.labels.append(np.sqrt(np.mean(npf3[1:7]**2)))
            self.num_classes = 1
        elif (self.lbl_selection == 'ef123_npf_TRMS_RRMS'):
            ef1_npf = np.load(self.root_dir / str(idx) / "label" / "npf1.npy").astype(float)
            ef2_npf = np.load(self.root_dir / str(idx) / "label" / "npf2.npy").astype(float)
            ef3_npf = np.load(self.root_dir / str(idx) / "label" / "npf3.npy").astype(float)
            self.lbl_header = [ 'lambda1',
                                'NPF1 TRMS [1]',
                                'NPF1 RRMS [1]',
                                'lambda2',
                                'NPF2 TRMS [1]',
                                'NPF2 RRMS [1]',
                                'lambda3',
                                'NPF3 TRMS [1]',
                                'NPF3 RRMS [1]']
            def npf_to_RMS(npf):
                return np.sqrt(np.mean(npf**2))
            self.labels.append(np.array( [  ef1_npf[0],
                                            npf_to_RMS(ef1_npf[1:4]),
                                            npf_to_RMS(ef1_npf[4:7]),
                                            ef2_npf[0],
                                            npf_to_RMS(ef2_npf[1:4]),
                                            npf_to_RMS(ef2_npf[4:7]),
                                            ef3_npf[0],
                                            npf_to_RMS(ef3_npf[1:4]),
                                            npf_to_RMS(ef3_npf[4:7])
                                        ]))
            self.num_classes = 3 * 3
        elif (self.lbl_selection == 'ef123_npf_RMS'):
            ef1_npf = np.load(self.root_dir / str(idx) / "label" / "npf1.npy").astype(float)
            ef2_npf = np.load(self.root_dir / str(idx) / "label" / "npf2.npy").astype(float)
            ef3_npf = np.load(self.root_dir / str(idx) / "label" / "npf3.npy").astype(float)
            self.lbl_header = [ 'lambda1',
                                'NPF1 RMS [1]',
                                'lambda2',
                                'NPF2 RMS [1]',
                                'lambda3',
                                'NPF3 RMS [1]',
                                ]
            def npf_to_RMS(npf):
                return np.sqrt(np.mean(npf**2))
            self.labels.append(np.array([  ef1_npf[0],
                                            npf_to_RMS(ef1_npf[1:7]),
                                            ef2_npf[0],
                                            npf_to_RMS(ef2_npf[1:7]),
                                            ef3_npf[0],
                                            npf_to_RMS(ef3_npf[1:7])
                                        ]))
            self.num_classes = 2 * 3
        elif (self.lbl_selection == 'npf123_RMS'):
            ef1_npf = np.load(self.root_dir / str(idx) / "label" / "npf1.npy").astype(float)
            ef2_npf = np.load(self.root_dir / str(idx) / "label" / "npf2.npy").astype(float)
            ef3_npf = np.load(self.root_dir / str(idx) / "label" / "npf3.npy").astype(float)
            self.lbl_header = [ 'NPF1 RMS [1]',
                                'NPF2 RMS [1]',
                                'NPF3 RMS [1]'
                                ]
            def npf_to_RMS(npf):
                return np.sqrt(np.mean(npf**2))
            self.labels.append(np.array([   npf_to_RMS(ef1_npf[1:7]),
                                            npf_to_RMS(ef2_npf[1:7]),
                                            npf_to_RMS(ef3_npf[1:7])
                                        ]))
            self.num_classes = 3
        elif (self.lbl_selection == 'npf1_MEAN'):
            npf1 = np.load(self.root_dir / str(idx) / "label" / "npf1.npy")
            self.lbl_header = ['npf1_MEAN']
            self.labels.append(np.mean(npf1.astype(float)[1:7]))
            self.num_classes = 1
        elif (self.lbl_selection == 'npf1_SUM'):
            npf1 = np.load(self.root_dir / str(idx) / "label" / "npf1.npy")
            self.lbl_header = ['npf1_SUM']
            self.labels.append(np.sum(npf1.astype(float)[1:7]))
            self.num_classes = 1
        elif (self.lbl_selection == 'npf1_MAX'):
            npf1 = np.load(self.root_dir / str(idx) / "label" / "npf1.npy")
            self.lbl_header = ['npf1_MAX']
            self.labels.append(np.max(npf1.astype(float)[1:7]))
            self.num_classes = 1
        else:
            print("ERROR: Unknown parameter for self.lbl_selection: ", self.lbl_selection)
            self.labels.append(np.array(0))
    def _split_data(self):
        # segments data according to self.split
        if self.split.size == 2:
            self.sep_idx = int(max(self.datafolders)*self.split[0])
            if self.set_type == 'train':
                self.folder_range = range(0, self.sep_idx)
            elif self.set_type == 'validation':
                self.folder_range = range(self.sep_idx, max(self.datafolders))
            elif self.set_type == 'test':
                print("Please select a 3-way set split to get a test set.")
                return -1
            else:
                print("Unexpected set_type: ", set_type)
                return -1
        elif self.split.size == 3:
            self.sep_idx = int(max(self.datafolders)*self.split[0])
            self.sep_idx_2 = self.sep_idx + int(max(self.datafolders)*self.split[1])
            if self.set_type == 'train':
                self.folder_range = range(0, self.sep_idx)
            elif self.set_type == 'validation':
                self.folder_range = range(self.sep_idx, self.sep_idx_2)
            elif self.set_type == 'test':
                self.folder_range = range(self.sep_idx_2, max(self.datafolders))

    def getLblAmax(self):
        # expect torch.tensor input
        return torch.max(self.labels, 0)[0]
    def getLblAmin(self):
        # expect torch.tensor input
        return torch.min(self.labels, 0)[0]
    def getLblStd(self):
        # expect torch.tensor input
        return torch.std(self.labels, 0)
    def getLblAvg(self):
        # expect torch.tensor input
        return torch.mean(self.labels, 0)
    def getLblHeader(self):
        return self.lbl_header
    def getNumClasses(self):
        return self.num_classes
    def logMask(self, mask = None):
        if mask is not None:
            for col in mask:
                print("Apply log to col ", col, " of label tensor ", self.labels.size())
                self.labels[:,col] = torch.log(self.labels[:,col])
        else:
            print("logMask: no mask provided! logMask() skipped.")
    def normalizeLabels(self, lower_bound = 0, upper_bound = 1,
                 lbl_min = None, lbl_max = None, avg = None, stdev = None, logmask = None, mode = 'pass_through'):
        '''
            A: Min value Dataset
            B: Max value Dataset
            a: lower_bound normalized data
            b: upper_bound normalized data

            x_norm = a + (x-A)*(b-a)/(B-A)

            inverse function
            x = A + (x_norm - a)*(B-A)/(b-a)
        '''
        if mode == 'pass_through':
            return
        epsilon = torch.tensor(1e-12, dtype = torch.float)
        if (mode == 'feature_scaling'):
            if lbl_min is None or lbl_max is None:
                print("normalize:{} requires lbl_min {} and lbl_max {} set.".format(mode, lbl_min, lbl_max)) 
            A = lbl_min
            B = lbl_max
            a = lower_bound
            b = upper_bound
            print("lbl_min: ", lbl_min, "\tlbl_max: ", lbl_max)
            print("a: ", a, "\tb: ", b)
            self.labels = a + (self.labels - A)*(b-a)/torch.max((B-A), epsilon)
            self.is_normalized = True
            if logmask is not None:
                print("Minimum value in labels (cannot be <0):",torch.min( self.labels))
                self.logMask(mask = logmask)
                print(torch.max(self.labels), " max value ", torch.min(self.labels), " min value")
        elif (mode == 'z-norm'):
            if avg is None or stdev is None:
                print("normalize:{} requires avg {} and stdev {} set.".format(mode, avg, stdev))
            self.labels = (self.labels - avg)/torch.max(stdev, epsilon)
            self.is_normalized = True
        else:
            print("Warning: mode ", mode, " is not implemented yet. Normalization skipped")

def denormalize(data, lower_bound = 0, upper_bound = 1,
                 lbl_min = None, lbl_max = None, avg = None, stdev = None, 
                 logmask = None, mode = 'pass_through'):
    '''
        A: Min value Dataset
        B: Max value Dataset
        a: lower_bound normalized data
        b: upper_bound normalized data

        x_norm = a + (x-A)*(b-a)/(B-A)

        inverse function
        x = A + (x_norm - a)*(B-A)/(b-a)
    '''
    # convert torch tensor to numpy for computation
    if mode == 'pass_through':
        return data
    epsilon = torch.tensor(1e-12, dtype = torch.float)
    #if not "torch.Tensor" in str(type(data)):
    #    data = torch.from_numpy(np.array(data))
    if (mode == 'feature_scaling'):
        if lbl_min is None or lbl_max is None:
            print("denormalize:{} requires lbl_min {} and lbl_max {} set.".format(mode, lbl_min, lbl_max))
        A = lbl_min
        B = lbl_max
        a = torch.tensor(lower_bound, dtype = torch.float)
        b = torch.tensor(upper_bound, dtype = torch.float) 
        #self.is_normalized = False (no access to self, what to do?)
        if logmask is not None:
            data = reverseLogMask(data, mask = logmask)
        return A + (data - a)*(B-A)/torch.max((b-a), epsilon)
    elif (mode == 'z-norm'):
        if avg is None or stdev is None:
            print("normalize:{} requires avg {} and stdev {} set.".format(mode, avg, stdev))
        #self.is_normalized = False (no access to self, what to do?)
        return data*torch.max(stdev,epsilon) + avg
    else:
        print("Warning: mode ", mode, " is not implemented yet. Denormalization skipped")


def reverseLogMask(data, mask):
    # expects torch tensor
    if mask is not None:
        for col in mask:
            #print("Apply exp to col ", col, " of label tensor ", data.size())
            data[:,col] = torch.exp(data[:,col])
        return data
    else:
        print("logMask: no mask provided! logMask() skipped.")
        return data

'''
Transformation Classes:
    Implement transformations as classes to facilitate usage.
    Note: All other classes rescale and tograyscale were deleted (reimplement if needed)
'''
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, lbl, extrude_length = sample['img'], sample['lbl'], sample['extrude_length']
        volume, verts, load_case = sample['volume'], sample['verts'], sample['load_case'] 
        # swap color axis
        # numpy image: H x W x C
        # torch image: C x H x w
        image = image.transpose((0, 1))
        img_dim = image.shape
        #image = image[1,:,:]
        return {'img': torch.from_numpy(image).float().reshape(1,img_dim[0],img_dim[1]),
                'lbl': torch.from_numpy(np.asarray(lbl)).float(),
                'extrude_length': extrude_length,
                'volume': volume,
                'verts': verts, 'load_case': load_case }

