#!/usr/bin/env python #
"""\
This script runs the simulation of the droplet evaporation. 

Usage: ./initialise or python initialise.py

"""

from __future__ import division
import physics_methods as pm
import io_methods as iom
import visualisation_methods as vm
import numpy as np
import simulate as mod
import os
import time
import importlib
import sys

def create_initial_conditions(val, directory):
    """Function for writing dictionary initial conditions for Wray or Masoud 
    scripts."""
    print("input directory: ",directory)
    c = iom.read_config(directory, 'DTM_config')
    
    if c.filter_touching == True:
        c.CX, c.CY, c.Rb, c.CA = pm.TouchingCircles(c.CX,c.CY,c.Rb,c.CA)

    filename = c.prefix+'_'+str(len(c.CX))
    RunTimeInputs={} # defines dict
    RunTimeInputs['Rb']=c.Rb                 # Droplet base radius (metres)
    RunTimeInputs['dt']= c.dt                # single number (s)
    RunTimeInputs['CA']=c.CA                 # as 1D np array (rads)
    RunTimeInputs['xcentres']=c.CX           # as 1D np array ()
    RunTimeInputs['ycentres']=c.CY           # as 1D np array
    RunTimeInputs['DNum']=len(c.CX)            # total number of droplets in array
    RunTimeInputs['Ambient_RHs']=c.Ambient_RHs # Fraction [0-1]
    RunTimeInputs['Ambient_T']=c.Ambient_T     # Degrees C
    RunTimeInputs['t']=c.t                     # initial time (s)
    RunTimeInputs['model']=c.model             # which model to simulate with "Wray" or "Masoud"
    RunTimeInputs['Directory']=directory       # where to save data (absolute, no trailing \)
    RunTimeInputs['Filename']=filename         # what to call data (no exts)
    RunTimeInputs['Transient_Length']=c.TL     # delay before updating evap rate (s)
    RunTimeInputs['bias_point']= c.bp          # xy coords of last point in array to evaporate 1D np array
    RunTimeInputs['bias_grad'] = c.bg          # magnitude of bias
    RunTimeInputs['nterms'] = c.nterms         # number of terms in Masoud max = 2
    RunTimeInputs['mode']=c.mode               # evaporation mode CCA or CCR
    RunTimeInputs['rand']=c.rand               # Randomise evaporation rates 
    RunTimeInputs['p_rate']=c.p_rate           # rate at which droplets are printed (s-1), 0 is instantaneous
    RunTimeInputs['t_terminate']=c.t_terminate # max simulation time (s)
    RunTimeInputs['n_mols'] = c.n_mols         # mols of solute in liquid
    RunTimeInputs['i'] = c.i                   # Van hoff factor


    if c.D=="water":
        RunTimeInputs['D']=pm.diffusion_coeff(c.Ambient_T)
    else:
        RunTimeInputs['D']=c.D
    if set(['sort']).issubset(dir(c)):
        RunTimeInputs['sort']=c.sort
    RunTimeInputs['Antoine_coeffs'] = c.ABCs
    RunTimeInputs['box_volume'] = c.box_volume
    RunTimeInputs['rho_liquid'] = c.rho_liquid
    RunTimeInputs['molar_masses'] = c.molar_masses
    RunTimeInputs['surface_tension'] = c.surface_tension
    RunTimeInputs['Vi'] = pm.GetVolumeCA(RunTimeInputs['CA'],RunTimeInputs['Rb'])
    print("Total volume= ", np.sum(RunTimeInputs['Vi']), " (L)")
    
    return RunTimeInputs, c.saving, c.live_plot, c.compare, c.cmap_name, c.dpi, c.vid_FPS, c.export_nframes

directory = input('Enter a file path: ')

for i in range(1):
    RunTimeInputs, saving, live_plot, compare, cmap_name, set_dpi ,set_FPS,export_nframes = create_initial_conditions(i,directory)
    
    tic = time.perf_counter()
     
    out_name = RunTimeInputs['model']+RunTimeInputs['Filename']
    out_target  = os.path.join(directory,out_name)
    iom.pickle_dict(directory, out_name+"_IC", RunTimeInputs) # save initialisation data for continued simulations
    RunTimeInputs = mod.Iterate(RunTimeInputs, out_target, live_plot)
    toc = time.perf_counter()
    # Results['simulation_time']=toc-tic    

    if saving:
        vm.ReportResults(out_target, RunTimeInputs, cmap_name)
        name = os.path.join(directory,'video')
        vm.export_video(out_target, RunTimeInputs, number_of_frames=export_nframes, odpi=set_dpi, vid_FPS = set_FPS, cmap_name='jet')
    if compare:
        eResults = iom.load_MDL_pickle(RunTimeInputs['Directory'])
        if 'sort' in list(RunTimeInputs.keys()):
            eResults['X'] = eResults['X'][RunTimeInputs['sort']]
            eResults['Y'] = eResults['Y'][RunTimeInputs['sort']]
            eResults['drying_times'] = eResults['drying_times'][RunTimeInputs['sort']]
            eResults['rdx'] = eResults['rdx'][RunTimeInputs['sort']]

        vm.Compare2Data(eResults, cmap_name) # fix for new data handling
        # print("bias angle  = ",Results['bias_angle'])
        # print("bias gradient  = ",Results['bias_grad'])
        
    
# =============================================================================