#!/usr/bin/env python #
"""\
This file contains input/output functions used for file handling by the code. 

"""

from __future__ import division
import numpy as np
from PIL import ImageColor
import os
import sys
import importlib
import glob
import pickle
import xml.etree.ElementTree as ET

def read_svg_circles(svg_path):
    tree = ET.parse(svg_path)
    root = tree.getroot()
    
    # These lists will store the cx, cy, and r values
    cx_vals = []
    cy_vals = []
    r_vals = []
    
    # Find all <circle> elements in the SVG
    for circle in root.iter('{http://www.w3.org/2000/svg}circle'):
        cx_vals.append(float(circle.get('cx')))
        cy_vals.append(float(circle.get('cy')))
        r_vals.append(float(circle.get('r')))
    
    return cx_vals, cy_vals, r_vals


def ReadSVGFile(directory, filename):
    """Read optimised SVG text file and extracts circle coordinates
    for input into Wray_et_al.2020.py.
    Input: "Filename".svg 
    Returns: Droplet Base Radius, Contact angle, X and Y droplet 
                centres."""
    CX    = np.empty((0,1), float)
    CY    = np.empty((0,1), float)
    R     = np.empty((0,1), float)
    CA    = np.empty((0,1), float)


    filename=filename+".svg"
    file=open(os.path.join(directory,filename), 'r')
    print("Found file: ", filename)
    print("at: ", directory)
    txt=file.readlines()
    ca_global=[0]
    for loop in range(len(txt)):

            if txt[loop].find("<g fill")!=-1:

                #print("All the droplets have the same contact angle.")
                fillL=txt[loop].find("fill=\"")
                fillU=txt[loop].find("\"",fillL+6)
                fill=txt[loop][fillL+6  :fillU]
                RGB=np.asarray(ImageColor.getcolor(fill, "RGB")) # convert hex to RGB
                ca_global=(np.pi/255)*RGB # Radians
            
            elif(txt[loop].find("<circle")==-1 and txt[loop].find("fill=\"")!=-1): # check all non circle lines for global CA
                #print("looking for fill in non circle line")
                fillL=txt[loop].find("fill=\"")
                fillU=txt[loop].find("\"",fillL+6)
                fill=txt[loop][fillL+6:fillU]
                RGB=np.asarray(ImageColor.getcolor(fill, "RGB"))
                ca_global=(np.pi/255)*RGB # Radians



            elif(txt[loop].find("<circle"))!=-1:
                #print("Circle line found")
                # Extract droplet x position    
                cxL=txt[loop].find("cx=\"")
                cxU=txt[loop].find("\"",cxL+4)
                cx=np.double(txt[loop][cxL+4:cxU])/1e3
                CX = np.append(CX, abs(cx)) 

                # Extract droplet y position
                cyL=txt[loop].find("cy=\"")
                cyU=txt[loop].find("\"",cyL+4)
                cy= np.double(txt[loop][cyL+4:cyU])/1e3
                CY = np.append(CY, abs(cy)) # 

                # Extract droplet radius
                rL=txt[loop].find("r=\"")
                rU=txt[loop].find(" ",rL)

                r=np.double(txt[loop][rL+3:rU-1])/1000

                R = np.append(R, r) # 
                #format_float = "{:.10f}".format(r)
                
                # Extract droplet ca
                fillL=txt[loop].find("fill=\"")
                if fillL != -1: # Check anyway for individual CA
                    fillU=txt[loop].find("\"",fillL+6)
                    fill=txt[loop][fillL+6:fillU]
                    RGB=np.asarray(ImageColor.getcolor(fill, "RGB"))
                    ca_current=(np.pi/255)*RGB # Radians
                    CA = np.append(CA, ca_current[0]) # 
                else: # write global CA if individual CA not found
                    CA = np.append(CA, ca_global[0]) #

            else:
                print("All detection options missed")
    CY=-CY+2*max(CY)
    file.close()
    print("_____________________________________________")
    print("Values retrieved from InkScape")
    print("_____________________________________________")
    print("\tCX= ",CX)
    print("\tCY= ",CY)
    print("\tCA= ",CA)
    print("\tR=  ",R)
    return CX, CY, R, CA

def read_config(config_dir, file):
    sys.path.insert(0, config_dir)
    config=importlib.import_module(file, package=None) 
    return config

def load_MDL_pickle(directory, prefix=None):
    """Load a measure_droplet_lifetime pickle.
        prefix (Optional): specify a specific filename (no ext)."""
    #print("Trying with .pickle")
    if prefix==None:
        prefix = "MDL_*.pickle"
    else:
        prefix = prefix + ".pickle"
    try:
        target = glob.glob(os.path.join(directory,prefix))[0] # with .pickle
        if os.path.getsize(target)>0:
            with open(target, 'rb') as handle:
                input_dict = pickle.load(handle)
    except:
        #print("Trying .pkl")
        if prefix==None:
            prefix = "MDL_*.pkl"
        else:
            prefix = prefix[:4]+ ".pkl"
        target = glob.glob(os.path.join(directory,prefix))[0] # with .pkl
        if os.path.getsize(target)>0:
            with open(target, 'rb') as handle:
                input_dict = pickle.load(handle)

    return input_dict

def load_MDTM_pickle(directory, prefix=None):
    """Load a droplet theory prediction pickle.
        prefix (Optional): specify a specific filename (no ext)."""
    #print("Trying with .pickle")
    if prefix==None:
        prefix1 = "MDTM_*.pkl"
    else:
        prefix1 = prefix + ".pkl"
    try:
        target = glob.glob(os.path.join(directory,prefix1))[0]
        if os.path.getsize(target)>0:
            with open(target, 'rb') as handle:
                input_dict = pickle.load(handle)
    except:
        #print("Trying .pkl")   
        if prefix==None:
            prefix = "MDTM*.pkl"
        else:
            prefix = prefix[:4]+ ".pkl"
        target = glob.glob(os.path.join(directory,prefix))[0]        
        if os.path.getsize(target)>0:
            with open(target, 'rb') as handle:
                input_dict = pickle.load(handle)

    return input_dict


def pickle_dict(export_directory, export_name, pickle_file):
    Resultsfile = open(os.path.join(export_directory, export_name+'.pkl'), 'wb')
    pickle.dump(pickle_file, Resultsfile)
    Resultsfile.close()
    return 




def read_imageJ_coords(directory, filename):
    """Reads ImageJ (FIJI) droplet centre coords."""
    data = np.genfromtxt(os.path.join(directory,filename+".txt"), dtype =(int, int, int, int), skip_header=0, names=True, usecols = (0,1,2,3))
    #data['Y']=im_h-data['Y']
    data['BX']=data['BX']+(data['Width']/2) # convert from upper left to centre coord
    data['BY']=data['BY']+(data['Height']/2) # convert from upper left to centre coord
    XY = np.array((data['BX'],data['BY']))
    WH = np.array((data['Width'],data['Height']))
    dr = np.mean(WH,axis=0)/2
    rs = np.min(XY, axis=0)/2
    print("1. Droplet centres retrieved")
    return data, XY, rs, dr