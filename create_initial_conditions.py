from __future__ import division
import droppy as dpy
import numpy as np
import The_Models as mod
import sys
sys.path.insert(0, r'\\NTU-DPM-CTY.ads.ntu.ac.uk\ERD160_projects$\aaaa_Joey\Scripts')
import Visualisation as vis

def create_initial_conditions(val, directory):
    """Function for writing dictionary initial conditions for Wray or Masoud 
    scripts."""
    c = dpy.read_config(directory, "MDTM_config")
    if c.filter_touching == True:
        c.CX, c.CY, c.Rb, c.CA = dpy.TouchingCircles(c.c.CX,c.CY,c.Rb,c.CA)
        
    filename = c.prefix+'_'+str(len(c.CX))
    RunTimeInputs={} # defines dict
    RunTimeInputs['Rb']=c.Rb                 # Droplet base radius (metres)
    RunTimeInputs['dt']= c.dt                # single number (s)
    RunTimeInputs['CA']=c.CA                 # as 1D np array (rads)
    RunTimeInputs['xcentres']=c.CX           # as 1D np array ()
    RunTimeInputs['ycentres']=c.CY           # as 1D np array
    RunTimeInputs['DNum']=len(c.CX)          # total number of droplets in array
    RunTimeInputs['Ambient_RH']=c.Ambient_RH # Fraction [0-1]
    RunTimeInputs['Ambient_T']=c.Ambient_T   # Degrees C
    RunTimeInputs['t']=c.t                   # initial time (s)
    RunTimeInputs['model']=c.model           # which model to simulate with "Wray" or "Masoud"
    RunTimeInputs['Directory']=directory     # where to save data (absolute, no trailing \)
    RunTimeInputs['Filename']=filename       # what to call data (no exts)
    RunTimeInputs['Transient_Length']=c.TL   # delay before updating evap rate (s)
    RunTimeInputs['Vi'] = dpy.GetVolumeCA(RunTimeInputs['CA'],RunTimeInputs['Rb'])
    
    return RunTimeInputs

# ==================USER INPUTS============================================

directory = r'Z:\aaaa_Joey\Large Arrays\Convection\Temperature_variations'
saving = True
compare = False

# config file:
    # saving : 
    # compare : 
    # filter_touching : 
    # prefix  : 
    # CX,CY,CA,Rb :
    # #CX, CY, Rb = 
    # Ambient_RH, Ambient_T, t, dt, TL = 0.30, 25, 0., 500, 0
    # model = "Masoud" # "Wray" or "Masoud"
#==========================================================================

for i in range(1):
    RunTimeInputs = create_initial_conditions(i,directory)
    if RunTimeInputs['model'] == "Wray":
        Results = mod.WrayEvaporate(RunTimeInputs)
    elif RunTimeInputs['model'] == "Masoud":
        Results = mod.MasoudEvaporate(RunTimeInputs)
    else:
        print("\nInvalid model name.")
        break
    if saving:
        dpy.ReportResults(Results)
    if compare:
        eResults = dpy.load_MDL_pickle(RunTimeInputs['Directory'])
        vis.Compare2Data(Results, eResults)
    
# =============================================================================
#     
# def Initialise_breath():
#     """."""
#     # SVG="5x2 Rectangle"
#     # directory=r'Z:\aaaa_Joey\Experiments\Kieran\5x2Array'
# 
#     # SVG="Optimised_7_Droplet_Hexagon_easy"
#     # directory=r'Z:\aaaa_Joey\Experiments\PD-7D-H-3'
#     #CX, CY, Rb, CA = dpy.ReadSVGFile(directory, SVG)
#     
#     # CX = np.genfromtxt(os.path.join(directory,'CX.txt'), delimiter=',')/1000
#     # CY = np.genfromtxt(os.path.join(directory,'CY.txt'), delimiter=',')/1000
#     # Rb = np.genfromtxt(os.path.join(directory,'Rb.txt'), delimiter=',')/1000
#     # CA = dpy.genfromtxt(os.path.join(directory,'CA.txt'), delimiter=',')*pi/180
#     #ca = np.ones(len(cx))*np.pi/2 
#     
#          
#     # CX_unit=cx[touching]-min(cx[touching]) # place bottom corner on origin
#     # CY_unit=cy[touching]-min(cy[touching]) # place bottom corner on origin
#     # Rb_unit=r[touching]
#     # CA_unit=ca[touching] 
#     # CX=CX_unit
#     # CY=CY_unit
#     # Rb=Rb_unit
#     # CA=CA_unit
#     # D=1
#     # i,j=indices((D,D))
#     # xshift=max(CX)
#     # yshift=max(CY)
#     # # mirror unit breath over NXN grid
#     # print("Mirroring")
#     # for idx in range(D):
#     #     for jdx in range(D):
#     #         print(len(CX_unit))
#     #         if idx ==0 and jdx ==0:
#     #             print("Skipping iteration")
#     #             continue
#     #         print("Adding more droplets")
#     #         CX_new=CX_unit + i[idx,jdx]*xshift 
#     #         CY_new=CY_unit + j[idx,jdx]*yshift
#     #         CX=np.append(CX,CX_new)
#     #         CY=np.append(CY,CY_new)
#     #         Rb=np.append(Rb,Rb_unit)
#     #         CA=np.append(CA,CA_unit)
# # =============================================================================
# #     filter_touching=True
# #     plt.close('all')
# #     if filter_touching == True:
# #         CX, CY, Rb, CA = TouchingCircles(cx,cy,r,ca)
# #         
# #     filename = prefix+'_'+str(len(CX))
# #     RunTimeInputs={} # defines dict
# #     RunTimeInputs['Rb']=Rb                 # Droplet base radius (metres)
# #     RunTimeInputs['dt']= dt
# #     RunTimeInputs['CA']=CA
# #     RunTimeInputs['xcentres']=CX           # as 1D np array
# #     RunTimeInputs['ycentres']=CY           # as 1D np array
# #     RunTimeInputs['DNum']=len(CX)          # total number of droplets in array
# #     RunTimeInputs['Ambient_RH']=Ambient_RH # Fraction [0-1]
# #     RunTimeInputs['Ambient_T']=Ambient_T   # Degrees C
# #     RunTimeInputs['t']=t                   # initial time (s)
# #     RunTimeInputs['directory']=directory   # where to save data (absolute, no trailing \)
# #     RunTimeInputs['filename']=filename     # what to call data (no exts)
# #     RunTimeInputs['Vi'] = GetVolumeCA(RunTimeInputs['CA'],RunTimeInputs['Rb'])
# # =============================================================================
#     return
#    
# 
# =============================================================================
