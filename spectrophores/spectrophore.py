# -*- encoding: utf-8 -*-
# author: Fábio Mendes dos Santos
# date: 04-02-2016

#RDKit library
from __future__ import print_function
from optparse import OptionParser
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols

import numpy as np
import scipy
import scipy.linalg
import math

class spectrophoreCalculator:
    #########################
    # Instance initalization
    #########################
    def __init__(self, resolution=3.0, accuracy=20, stereo='none', normalization='none'):
        # Initiate resolution
        if resolution > 0: 
            self.resolution = resolution
        else: 
            raise ValueError('Resolution should be larger than 0')
        
        # Initiate accuracy
        self.accuracy = int(accuracy)
        if   self.accuracy == 1: self.rotationStepList = (1,)
        elif self.accuracy == 2: self.rotationStepList = (2, 5, 36)
        elif self.accuracy == 5: self.rotationStepList = (5, 36)
        elif self.accuracy == 10: self.rotationStepList = (10, 15, 36)
        elif self.accuracy == 15: self.rotationStepList = (15, 20, 36)
        elif self.accuracy == 20: self.rotationStepList = (20, 30, 36, 45)
        elif self.accuracy == 30: self.rotationStepList = (30, 36, 45)
        elif self.accuracy == 36: self.rotationStepList = (36, 45, 60)
        elif self.accuracy == 45: self.rotationStepList = (45, 60)
        elif self.accuracy == 60: self.rotationStepList = (60,)
        else: raise ValueError(
            'Accuracy should be 1, 2, 5, 10, 15, 20, 30, 36, 45 or 60')

        # Initiate stereo
        self.stereo = stereo.lower() #lowercase
        if self.stereo == 'none':
            self.beginProbe = 0
            self.endProbe = 12
            self.numberOfProbes = 12
        elif self.stereo == 'unique':
            self.beginProbe = 12
            self.endProbe = 30
            self.numberOfProbes = 18
        elif self.stereo == 'mirror':
            self.beginProbe = 30
            self.endProbe = 48
            self.numberOfProbes = 18
        elif self.stereo == 'all':
            self.beginProbe = 12
            self.endProbe = 48
            self.numberOfProbes = 36
        else: raise ValueError(
            'The stereo flag should be "none", "unique", "mirror" or "all"')

        # Initiate the type of normalization
        if normalization.lower() in ('none', 'mean', 'std', 'all'): 
            self.normalization = normalization.lower()
        else: raise ValueError(
            'The normalization flag should be "none", "mean", "std" or "all"')

        # Set up some Spectrophore variables
        self.numberOfProperties = 4
        self.sphoreSize = self.numberOfProperties * self.numberOfProbes
        self.sphore = np.zeros(self.sphoreSize)
        self.energy = np.zeros(self.sphoreSize)
        self.numberOfBoxpoints = 12
        
        # Set up the box
        self.probe = np.array([
            #  1 / Dodecapole - non-stereo - probe 1
            [+1, +1, -1, -1, -1, +1, +1, -1, -1, -1, +1, +1],
            #  2 / Dodecapole - non-stereo - probe 2
            [+1, +1, -1, -1, +1, -1, -1, +1, -1, -1, +1, +1], 
            #  3 / Dodecapole - non-stereo - probe 3
            [+1, +1, +1, -1, -1, -1, -1, -1, -1, +1, +1, +1], 
            #  4 / Dodecapole - non-stereo - probe 4
            [+1, +1, +1, -1, -1, -1, -1, -1, +1, +1, -1, +1], 
            #  5 / Dodecapole - non-stereo - probe 5
            [+1, +1, +1, -1, -1, +1, -1, +1, -1, -1, +1, -1], 
            #  6 / Dodecapole - non-stereo - probe 6
            [+1, +1, +1, -1, +1, -1, +1, -1, -1, -1, +1, -1], 
            #  7 / Dodecapole - non-stereo - probe 7
            [+1, +1, +1, -1, +1, -1, +1, -1, +1, -1, -1, -1], 
            #  8 / Dodecapole - non-stereo - probe 8
            [+1, +1, +1, +1, -1, -1, -1, -1, +1, -1, +1, -1], 
            #  9 / Dodecapole - non-stereo - probe 9
            [+1, +1, +1, +1, -1, -1, -1, -1, +1, +1, -1, -1], 
            # 10 / Dodecapole - non-stereo - probe 10
            [+1, +1, +1, +1, +1, -1, -1, +1, -1, -1, -1, -1], 
            # 11 / Dodecapole - non-stereo - probe 11
            [+1, +1, +1, +1, +1, +1, -1, -1, -1, -1, -1, -1], 
            # 12 / Dodecapole - non-stereo - probe 12
            [+1, +1, +1, -1, -1, +1, -1, -1, -1, +1, -1, +1], 
            # 13 / Dodecapole - mirror-stereo - probe 1
            [+1, +1, -1, -1, -1, -1, +1, +1, -1, +1, +1, -1], 
            # 14 / Dodecapole - mirror-stereo - probe 2
            [+1, +1, +1, -1, -1, -1, -1, -1, +1, -1, +1, +1], 
            # 15 / Dodecapole - mirror-stereo - probe 3
            [+1, +1, +1, -1, -1, -1, -1, +1, -1, +1, +1, -1], 
            # 16 / Dodecapole - mirror-stereo - probe 4
            [+1, +1, +1, -1, -1, -1, +1, -1, -1, +1, -1, +1], 
            # 17 / Dodecapole - mirror-stereo - probe 5
            [+1, +1, +1, -1, -1, -1, +1, -1, -1, +1, +1, -1], 
            # 18 / Dodecapole - mirror-stereo - probe 6
            [+1, +1, +1, -1, -1, -1, +1, -1, +1, -1, +1, -1], 
            # 19 / Dodecapole - mirror-stereo - probe 7
            [+1, +1, +1, -1, -1, -1, +1, -1, +1, +1, -1, -1], 
            # 20 / Dodecapole - mirror-stereo - probe 8
            [+1, +1, +1, -1, -1, -1, +1, +1, -1, +1, -1, -1], 
            # 21 / Dodecapole - mirror-stereo - probe 9
            [+1, +1, +1, -1, -1, -1, +1, +1, +1, -1, -1, -1], 
            # 22 / Dodecapole - mirror-stereo - probe 10
            [+1, +1, +1, -1, -1, +1, +1, -1, -1, -1, -1, +1], 
            # 23 / Dodecapole - mirror-stereo - probe 11
            [+1, +1, +1, -1, -1, +1, +1, -1, -1, -1, +1, -1], 
            # 24 / Dodecapole - mirror-stereo - probe 12
            [+1, +1, +1, -1, -1, +1, +1, -1, -1, +1, -1, -1], 
            # 25 / Dodecapole - mirror-stereo - probe 13
            [+1, +1, +1, -1, -1, +1, +1, +1, -1, -1, -1, -1], 
            # 26 / Dodecapole - mirror-stereo - probe 14
            [+1, +1, +1, -1, +1, -1, -1, +1, +1, -1, -1, -1], 
            # 27 / Dodecapole - mirror-stereo - probe 15
            [+1, +1, +1, -1, +1, -1, +1, -1, -1, -1, -1, +1], 
            # 28 / Dodecapole - mirror-stereo - probe 16
            [+1, +1, +1, -1, +1, -1, +1, +1, -1, -1, -1, -1], 
            # 29 / Dodecapole - mirror-stereo - probe 17
            [+1, +1, +1, +1, +1, -1, -1, -1, -1, -1, -1, +1], 
            # 30 / Dodecapole - mirror-stereo - probe 18
            [+1, +1, +1, +1, +1, -1, -1, -1, -1, -1, +1, -1], 
            # 31 / Dodecapole - unique-stereo - probe 1
            [+1, +1, -1, -1, +1, -1, +1, -1, +1, -1, -1, +1], 
            # 32 / Dodecapole - unique-stereo - probe 2
            [+1, +1, +1, -1, -1, -1, -1, -1, +1, +1, +1, -1], 
            # 33 / Dodecapole - unique-stereo - probe 3
            [+1, +1, +1, -1, -1, +1, -1, -1, -1, -1, +1, +1], 
            # 34 / Dodecapole - unique-stereo - probe 4
            [+1, +1, +1, -1, +1, -1, -1, -1, -1, +1, -1, +1], 
            # 35 / Dodecapole - unique-stereo - probe 5
            [+1, +1, +1, -1, +1, -1, -1, -1, -1, -1, +1, +1], 
            # 36 / Dodecapole - unique-stereo - probe 6
            [+1, +1, +1, -1, +1, -1, -1, -1, +1, -1, +1, -1], 
            # 37 / Dodecapole - unique-stereo - probe 7
            [+1, +1, +1, -1, +1, -1, -1, -1, +1, -1, -1, +1], 
            # 38 / Dodecapole - unique-stereo - probe 8
            [+1, +1, +1, -1, +1, +1, -1, -1, -1, -1, -1, +1], 
            # 39 / Dodecapole - unique-stereo - probe 9
            [+1, +1, +1, -1, +1, +1, -1, -1, +1, -1, -1, -1], 
            # 40 / Dodecapole - unique-stereo - probe 10
            [+1, +1, +1, -1, +1, -1, -1, +1, -1, +1, -1, -1], 
            # 41 / Dodecapole - unique-stereo - probe 11
            [+1, +1, +1, -1, +1, -1, -1, +1, -1, -1, +1, -1], 
            # 42 / Dodecapole - unique-stereo - probe 12
            [+1, +1, +1, -1, +1, -1, -1, +1, -1, -1, -1, +1], 
            # 43 / Dodecapole - unique-stereo - probe 13
            [+1, +1, +1, -1, +1, +1, -1, +1, -1, -1, -1, -1], 
            # 44 / Dodecapole - unique-stereo - probe 14
            [+1, +1, +1, -1, +1, +1, -1, -1, -1, -1, +1, -1], 
            # 45 / Dodecapole - unique-stereo - probe 15
            [+1, +1, +1, -1, +1, -1, +1, -1, -1, +1, -1, -1], 
            # 46 / Dodecapole - unique-stereo - probe 16
            [+1, +1, +1, -1, +1, +1, +1, -1, -1, -1, -1, -1], 
            # 47 / Dodecapole - unique-stereo - probe 17
            [+1, +1, +1, +1, +1, -1, -1, -1, +1, -1, -1, -1], 
            # 48 / Dodecapole - unique-stereo - probe 18
            [+1, +1, +1, +1, +1, -1, -1, -1, -1, +1, -1, -1]  
        ])
        self.boxpoint = np.zeros((self.numberOfBoxpoints, 
                                  self.numberOfProperties + 3))
    
    ########################
    # Resolution
    ########################
    def setResolution(self, resolution=3.0):
        if resolution > 0: self.resolution = resolution
        else: raise ValueError('Resolution should be larger than 0')
    
    def getResolution(self):
        return self.resolution   

    ########################
    # Accuracy
    ########################
    def setAccuracy(self, accuracy=20):
        self.accuracy = int(accuracy)
        if   self.accuracy == 1: self.rotationStepList = (1,)
        elif self.accuracy == 2: self.rotationStepList = (2, 5, 36)
        elif self.accuracy == 5: self.rotationStepList = (5, 36)
        elif self.accuracy == 10: self.rotationStepList = (10, 15, 36)
        elif self.accuracy == 15: self.rotationStepList = (15, 20, 36)
        elif self.accuracy == 20: self.rotationStepList = (20, 30, 36, 45)
        elif self.accuracy == 30: self.rotationStepList = (30, 36, 45)
        elif self.accuracy == 36: self.rotationStepList = (36, 45, 60)
        elif self.accuracy == 45: self.rotationStepList = (45, 60)
        elif self.accuracy == 60: self.rotationStepList = (60,)
        else: raise ValueError(
            'Accuracy should be one of 1, 2, 5, 10, 15, 20, 30, 36, 45 or 60')
    
    def getAccuracy(self):
        return self.accuracy 
    
    ########################
    # Stereo
    ########################
    def setStereo(self, stereo='none'):
        self.stereo = stereo.lower()
        if self.stereo == 'none':
            self.beginProbe = 0
            self.endProbe = 12
            self.numberOfProbes = 12
        elif self.stereo == 'unique':
            self.beginProbe = 12
            self.endProbe = 30
            self.numberOfProbes = 18
        elif self.stereo == 'mirror':
            self.beginProbe = 30
            self.endProbe = 48
            self.numberOfProbes = 18
        elif self.stereo == 'all':
            self.beginProbe = 12
            self.endProbe = 48
            self.numberOfProbes = 36
        else: raise ValueError(
            'The stereo flag should be "none", "unique", "mirror" or "all"')
        self.sphoreSize = self.numberOfProperties * self.numberOfProbes
        self.sphore = np.zeros(self.sphoreSize)
        self.energy = np.zeros(self.sphoreSize)
        
    def getStereo(self):
        return self.stereo
    
    ########################
    # Normalisation
    ########################
    def setNormalization(self, normalization='none'):
        if normalization.lower() in ('none', 'mean', 'std', 'all'): 
            self.normalization = normalization.lower()
        else: raise ValueError(
        'The normalization flag should be "none", "mean", "std" or "all"')
        
    def getNormalization(self):
        return self.normalization
        
    ########################
    # Calculate Spectrophore
    ########################
    def calculate(self, mol, confID=0):
        
        # Check number of atoms after adding the hydrogens
        nAtoms = mol.GetNumAtoms()
        if nAtoms < 3: 
            raise ValueError( '>=3 atoms are needed in molecule, only %d given' % (nAtoms))
        
        # Atomic properties
        # [0]: atomic partial charges -> conformation dependent
        # [1]: atomic lipophilicities
        # [2]: atomic shape deviations -> conformation dependent
        # [3]: atomic electrophilicities -> conformation dependent
        prop = np.zeros((nAtoms, self.numberOfProperties))
        chi = np.zeros(nAtoms)
        eta = np.zeros(nAtoms)
        A = np.zeros((nAtoms + 1, nAtoms + 1))
        B = np.zeros(nAtoms + 1)
        #print(Chem.MolToMolBlock(mol))
       
        radii = np.zeros(nAtoms)
        a = 0        
        for atom in mol.GetAtoms():
            n = atom.GetAtomicNum()
            if   n ==  1:   # H
                radii[a] = +1.20
                eta[a] = +0.65971
                chi[a] = +0.20606
                if atom.GetTotalValence():
                    neighbors = atom.GetNeighbors()
                    prop[a][1] = -0.018   # non-polar
                    for neighbor in neighbors:
                        an = neighbor.GetAtomicNum()
                        if an != 1 and an != 6:
                            prop[a][1] = -0.374   # polar
                            break
                else:
                    prop[a][1] = -0.175
            elif n ==  3:   # Li
                radii[a] = +1.82                
                eta[a] = +0.32966
                chi[a] = +0.36237
                prop[a][1] = -0.175
            elif n ==  5:   # B
                radii[a] = +2.00                
                eta[a] = +0.32966
                chi[a] = +0.32966
                prop[a][1] = -0.175
            elif n ==  6:   # C
                radii[a] = +1.70                
                eta[a] = +0.32966
                chi[a] = +0.36237
                prop[a][1] = +0.271
            elif n ==  7:   # N
                radii[a] = +1.55
                eta[a] = +0.34519
                chi[a] = +0.49279
                prop[a][1] = -0.137
            elif n ==  8:   # O
                radii[a] = +1.52
                eta[a] = +0.54428
                chi[a] = +0.73013
                prop[a][1] = -0.321
            elif n ==  9:   # F
                radii[a] = +1.47
                eta[a] = +0.72664
                chi[a] = +0.72052
                prop[a][1] = +0.217
            elif n == 11:   # Na
                radii[a] = +2.27
                eta[a] = +0.32966
                chi[a] = +0.36237
                prop[a][1] = -0.175
            elif n == 12:   # Mg
                radii[a] = +1.73
                eta[a] = +0.32966
                chi[a] = +0.36237
                prop[a][1] = -0.175
            elif n == 14:   # Si
                radii[a] = +2.10
                eta[a] = +0.32966
                chi[a] = +0.36237
                prop[a][1] = -0.175
            elif n == 15:   # P
                radii[a] = +1.80
                eta[a] = +0.32966
                chi[a] = +0.36237
                prop[a][1] = -0.175
            elif n == 16:   # S
                radii[a] = +1.80
                eta[a] = +0.20640
                chi[a] = +0.62020
                prop[a][1] = +0.385
            elif n == 17:   # Cl
                radii[a] = +1.75
                eta[a] = +0.32966
                chi[a] = +0.36237
                prop[a][1] = +0.632
            elif n == 19:   # K
                radii[a] = +2.75
                eta[a] = +0.32966
                chi[a] = +0.36237
                prop[a][1] = -0.175
            elif n == 20:   # Ca
                radii[a] = +2.00
                eta[a] = +0.32966
                chi[a] = +0.36237
                prop[a][1] = -0.175
            elif n == 26:   # Fe
                radii[a] = +1.10
                eta[a] = +0.32966
                chi[a] = +0.36237
                prop[a][1] = -0.175
            elif n == 29:   # Cu
                radii[a] = +1.40
                eta[a] = +0.32966
                chi[a] = +0.36237
                prop[a][1] = -0.175
            elif n == 30:   # Zn
                radii[a] = +1.39
                eta[a] = +0.32966
                chi[a] = +0.36237
                prop[a][1] = -0.175
            elif n == 35:   # Br
                radii[a] = +1.85
                eta[a] = +0.54554
                chi[a] = +0.70052
                prop[a][1] = +0.815
            elif n == 53:   # I
                radii[a] = +1.98
                eta[a] = +0.30664
                chi[a] = +0.68052
                prop[a][1] = +0.198
            else:
                radii[a] = +1.50
                eta[a] = +0.65971
                chi[a] = +0.20606
                prop[a][1] = -0.175
            a += 1
        
        # Conformers
        if mol.GetNumConformers() < confID + 1:
            raise ValueError(
                'At least %d conformation(s) should be present, %d found' % 
                (confID + 1, mol.GetNumConformers()))
        conf = mol.GetConformer(confID)
        oricoor = np.zeros((nAtoms, 3))
        coor = np.zeros((nAtoms, 3))
        ref1 = np.zeros((nAtoms, 3)) ##
        ref2 = np.zeros((nAtoms, 3)) ##
       
        # Coordinates
        for r in range(nAtoms):
            c = conf.GetAtomPosition(r)
            (oricoor[r][0], oricoor[r][1], oricoor[r][2]) = (c[0], c[1], c[2])
            A[r][r] = 2 * eta[r]
        
        # Complete A matrix
        for r in range(nAtoms):
            for i in range(r + 1, nAtoms):
                d =  (oricoor[r][0] - oricoor[i][0]) * \
                     (oricoor[r][0] - oricoor[i][0])
                d += (oricoor[r][1] - oricoor[i][1]) * \
                     (oricoor[r][1] - oricoor[i][1])
                d += (oricoor[r][2] - oricoor[i][2]) * \
                     (oricoor[r][2] - oricoor[i][2])
                A[r][i] = 0.529176 / np.sqrt(d)    # Angstrom to au
                A[i][r] = A[r][i]   
         
        # Property [0]: partial atomic charges
        for i in range(nAtoms):
            A[i][nAtoms] = -1
            A[nAtoms][i] = +1
            B[i] = -chi[i]
        A[nAtoms][nAtoms] = 0
        B[nAtoms] = Chem.GetFormalCharge(mol)

        X = scipy.linalg.solve(A, B)
        chi2 = X[nAtoms] * X[nAtoms]
        for a in range(nAtoms): prop[a][0] = X[a]            
        
        # Property [2]: atomic shape deviations
        coor = list(oricoor)
        coor = np.asarray(coor)  
        cog = np.average(coor, axis=0)
        d = (coor - cog) * (coor - cog)
        d = np.sqrt(d.sum(axis=1))
        avg_d = np.average(d)
        std_d = np.std(d)
        
        #prop[:,2] = (d - avg_d) * avg_d
        prop[:,2] = (d - avg_d) / std_d

        # Property [3]: atomic electrophilicities
        B = np.ones(nAtoms + 1)
        B[nAtoms] = 0
        for i in range(nAtoms):
            A[i][nAtoms] = 0
            A[nAtoms][i] = +1
        A[nAtoms][nAtoms] = -1
        X = scipy.linalg.solve(A, B)
        for a in range(nAtoms): prop[a][3] = X[a] * chi2

        # Orient molecule to its center of gravity and orient in standard way
        # 1) Center molecule around its center of gravity
        cog = np.average(oricoor, axis=0)
        oricoor -= cog    
        
        # 2) Determine atom that is furthest away from origin
        d = oricoor * oricoor
        d = np.sqrt(d.sum(axis=1))
        maxAtom = np.argmax(d)
        
        # 3) Rotate all atoms along z-axis
        angle = -np.arctan2(oricoor[maxAtom][1], oricoor[maxAtom][0])
        c = np.cos(angle)
        s = np.sin(angle)
        for i in range(nAtoms):
            x = c * oricoor[i][0] - s * oricoor[i][1]
            y = s * oricoor[i][0] + c * oricoor[i][1]
            oricoor[i][0] = x
            oricoor[i][1] = y
           
        # 4) Rotate all atoms along y-axis to place the maxAtom on z
        angle = -np.arctan2(oricoor[maxAtom][0], oricoor[maxAtom][2])
        c = np.cos(angle)
        s = np.sin(angle)
        for i in range(nAtoms):
            x = c * oricoor[i][0] + s * oricoor[i][2]
            z = c * oricoor[i][2] - s * oricoor[i][0]
            oricoor[i][0] = x
            oricoor[i][2] = z
            
        # 5) Center molecule again around its COG
        cog = np.average(oricoor, axis=0)
        oricoor -= cog 
        
        from . import CRotate
        
        
        spectrophore = CRotate.CRotate(self.rotationStepList, oricoor, prop, self.normalization, self.resolution, self.beginProbe, self.endProbe, radii)        
        spectrophore = np.asarray(spectrophore)
        
        return spectrophore
