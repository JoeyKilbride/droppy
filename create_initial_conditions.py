from __future__ import division
import droppy as dpy
import numpy as np
import The_Models as mod
import sys
import os
#sys.path.insert(0, r'\\NTU-DPM-CTY.ads.ntu.ac.uk\ERD160_projects$\aaaa_Joey\Scripts') # Windows
sys.path.insert(0, r'/Volumes/ERD160_projects$/aaaa_Joey/Scripts/') # Mac
print("path:", sys.path)

def create_initial_conditions(val, directory):
    """Function for writing dictionary initial conditions for Wray or Masoud 
    scripts."""
    c = dpy.read_config(directory, "DTM_config")
    if c.filter_touching == True:
        c.CX, c.CY, c.Rb, c.CA = dpy.TouchingCircles(c.CX,c.CY,c.Rb,c.CA)

    filename = c.prefix+'_'+str(len(c.CX))
    RunTimeInputs={} # defines dict
    RunTimeInputs['Rb']=c.Rb                 # Droplet base radius (metres)
    RunTimeInputs['dt']= c.dt                # single number (s)
    RunTimeInputs['CA']=c.CA                 # as 1D np array (rads)
    RunTimeInputs['xcentres']=c.CX           # as 1D np array ()
    RunTimeInputs['ycentres']=c.CY           # as 1D np array
    RunTimeInputs['DNum']=len(c.CX)          # total number of droplets in array
    RunTimeInputs['Ambient_RHs']=c.Ambient_RHs # Fraction [0-1]
    RunTimeInputs['Ambient_T']=c.Ambient_T   # Degrees C
    RunTimeInputs['t']=c.t                   # initial time (s)
    RunTimeInputs['model']=c.model           # which model to simulate with "Wray" or "Masoud"
    RunTimeInputs['Directory']=directory     # where to save data (absolute, no trailing \)
    RunTimeInputs['Filename']=filename       # what to call data (no exts)
    RunTimeInputs['Transient_Length']=c.TL   # delay before updating evap rate (s)
    RunTimeInputs['bias_point']= c.bp        # xy coords of last point in array to evaporate 1D np array
    RunTimeInputs['bias_grad'] = c.bg
    RunTimeInputs['nterms'] = c.nterms
    RunTimeInputs['mode']=c.mode
    RunTimeInputs['rand']=c.rand

    if c.D=="water":
        RunTimeInputs['D']=dpy.diffusion_coeff(c.Ambient_T)
    else:
        RunTimeInputs['D']=c.D
    if set(['sort']).issubset(dir(c)):
        RunTimeInputs['sort']=c.sort
    RunTimeInputs['Antoine_coeffs'] = c.ABCs
    RunTimeInputs['box_volume'] = c.box_volume
    RunTimeInputs['rho_liquid'] = c.rho_liquid
    RunTimeInputs['molar_masses'] = c.molar_masses
    RunTimeInputs['surface_tension'] = c.surface_tension
    RunTimeInputs['Vi'] = dpy.GetVolumeCA(RunTimeInputs['CA'],RunTimeInputs['Rb'])
    print("Total volume= ", np.sum(RunTimeInputs['Vi']), " (m3)")
    
    return RunTimeInputs, c.saving, c.compare, c.cmap_name, c.dpi, c.export_nframes

# ==================USER INPUTS============================================

directory = input('Enter a file path: ')
#saving = True
#compare = False

# You need to have a config file in the directory specified above
# with the following variables defined:
    # saving : Boolean (True/False), saving the output data to pickle and visualisation images.
    # compare : Boolean (True/False), compare to experimental data in the same directory
    # filter_touching : Boolean (True/False), remove droplets which overlap
    # prefix  : String name for file saving
    # Droplet array geometries (numpy arrays)
        # CX,CY : droplet centre coordinates
        # CA,Rb : droplet initial contact angles and base radii
    # Environment:
        # Ambient_RH : Ambient relative humidity (0-1)
        # Ambient_T : Ambient temperature (degrees C)
    # Simulation parameters
        # t  : Initial time (should always be zero, i should potentially remove this?) 
        # dt : Timestep (seconds)
        # TL : Transient time (seconds), how long to wait before disappearing droplets influence the
        #      evaporation rate of the others
        # model : "Wray" or "Masoud" (some features unavailable with Wray)
        # mode : "CCA" or "CCR" (Constant Contact Angle or Constant Contact Radius)
        # box_volume : volume of the box droplets are evaporating in. (=np.inf, for constant RH)
        # rho_liquid : liquid density (kg/m^3)
        # A,B,C : Antoine Coefficients for liquid
        # molar_mass : molar mass of the liquid (kg/mol)

for i in range(1):
    RunTimeInputs, saving, compare, cmap_name, set_dpi, export_nframes = create_initial_conditions(i,directory)
    Results = mod.MasoudEvaporate(RunTimeInputs)
    
    # if RunTimeInputs['model'] == "Wray":
    #     Results = mod.WrayEvaporate(RunTimeInputs)
    # elif RunTimeInputs['model'] == "Masoud":
    #     Results = mod.MasoudEvaporate(RunTimeInputs)
    # else:
        # print("\nInvalid model name.")
        # break
    if saving:
        dpy.ReportResults(Results, cmap_name, RunTimeInputs['model'])
        name = os.path.join(directory,'video')
        dpy.export_video(Results, number_of_frames=export_nframes, odpi=set_dpi, cmap_name='jet')
    if compare:
        eResults = dpy.load_MDL_pickle(RunTimeInputs['Directory'])
        if 'sort' in list(RunTimeInputs.keys()):
            eResults['X'] = eResults['X'][RunTimeInputs['sort']]
            eResults['Y'] = eResults['Y'][RunTimeInputs['sort']]
            eResults['drying_times'] = eResults['drying_times'][RunTimeInputs['sort']]
            eResults['rdx'] = eResults['rdx'][RunTimeInputs['sort']]

        dpy.Compare2Data(Results, eResults, cmap_name)
        print("bias angle  = ",Results['bias_angle'])
        print("bias gradient  = ",Results['bias_grad'])
        
    
# =============================================================================