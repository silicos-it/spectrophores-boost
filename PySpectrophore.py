#!/usr/bin/env python
# author: Fabio Mendes dos Santos
# date: 04-02-2016

#RDKit library
from __future__ import print_function
from optparse import OptionParser
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols

# python library
import numpy as np

#personal libraries
from spectrophores.spectrophore import spectrophoreCalculator

import scipy
import scipy.linalg
import math

def similarityCalculation (ref,  normalization, stereo, rotateSize, resolution):
    
    #preparing Docking
    #print (index, normalization, stereo, rotateSize, resolution)
    calculator = spectrophoreCalculator(accuracy=rotateSize)
    calculator.setStereo(stereo)
    calculator.setNormalization(normalization)
    calculator.setResolution(resolution)
  
    ref_fps        = [calculator.calculate(x) for x in ref]
    
    return  ref_fps
 
def __menu ():
    #----------- Menu -----------#
    argv = OptionParser()
    argv.add_option("-i", "--InputFile",           action = "store", dest = "file_input",        type ="string", help = "Store the name of the file to input data")
    argv.add_option("-o", "--OutFile",             action = "store", dest = "file_output",    type ="string", default= "out.txt",  help = "File for the output file")
    argv.add_option("-n", "--Normalization",  action = "store", dest = "norm",            type ="string", default="none", help = "Specifies the kind of normalization that should be performed Valid values are:   none (default), mean, std, all")
    argv.add_option("-s", "--Stereo",               action = "store", dest = "stereo",          type ="string", default="none", help = "Specifies the kind of cages that should be use. Valid values are: none (default), unique, mirror, all")
    argv.add_option("-t", "--RotateSize",         action = "store", dest = "rotateSize",    type ="int",       default= 20, help = "1, 2, 5, 10, 15, 20 (default), 30, 36, 45, 60")
    argv.add_option("-e", "--Resolution",         action = "store", dest = "resolution",    type ="float",    default=3.0, help = "  teste ") 

    #argv.add_option("-a", "--AgleCutoff", action = "store", dest = "angle_cutoff", type ="int", default = 0, help = "Maximum value for the angle permited. Values from 0 to 180")
    #argv.add_option("-v", "--verbose", action = "store_true", dest = "verbose", help = "Faz o programa mostrar mais informacao")
    
    (argumentos, otherCommands) = argv.parse_args()
    
    if argumentos.file_input == None:
        argv.error("Mandatory option -i  or -o not given\nUsage: PySpectrophore -i <file> -o <outFile> [options].\nFor more informations type: PySpectrophore --help\n")    
        
    return argumentos

if __name__ == "__main__":
    argumentos = __menu()
    
    # reading entry files
    ref = Chem.SDMolSupplier(argumentos.file_input , removeHs=False)
    ref = [x for x in ref if x is not None] 
    
    # Printing number of files
    #print("File has ", len(ref), "  molecules")

    spectrophore =  similarityCalculation(ref, argumentos.norm, argumentos.stereo, argumentos.rotateSize, argumentos.resolution)
    
##    for i in spectrophore:
##        print(i,)
    
    
    output = open(argumentos.file_output, 'w')
    output.write("File: %s\n" %(argumentos.file_input))
    output.write("Number_of_Molecules: %d mols\n" %(len(ref)))

    output.write("\nDataset_Spectrophores:\n")
    for i in range(0, len(spectrophore)):
        name = ref[i].GetProp("_Name")
        
        if name == "": 
            output.write("unknown%d\t" %(i) )
        else:
            output.write("%s\t" %(ref[i].GetProp("_Name")))
            
        for j in spectrophore[i]:
            output.write("%s\t" %(j))
        output.write("\n")



    
