#!/usr/bin/env python #
"""\
This file contains visualisation functions used to display the results of the simulations. 

"""
from __future__ import division
import physics_methods as pm
import io_methods as iom
import numpy as np
import matplotlib
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle
from matplotlib.axes._axes import _log as matplotlib_axes_logger
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os
import pickle
import cv2
from io import BytesIO
from PIL import Image
matplotlib_axes_logger.setLevel('ERROR')

def CreateDroplets(ax, fig, cmaptype, centres, r0, C, vmin, vmax, multiplot, colourbar, lims=0):
    cmap = plt.get_cmap(cmaptype)
    normcmap = colors.Normalize(vmin=vmin, vmax=vmax)
    #normcmap = matplotlib.colors.Normalize(vmin=np.min([C.min(),C.min()]), vmax=np.max([C.max(),C.max()]))
    # ax.set_aspect(1)
    
    lst=[]
    clr=[]
    for idx, centre in enumerate(centres): 
        eclr=cmap(normcmap(C[idx]))
        circle = Circle((centre[0],centre[1]), r0[idx])
        lst.append(circle)
        clr.append(eclr) 
    collection = PatchCollection(lst)
    collection.set_facecolor(clr)
    s = ax.add_collection(collection)
    if not(isinstance(lims, np.ndarray)):
        cs=list(zip(*centres))
        mr=max(r0)
        lims = np.array([[min(cs[0])-mr, max(cs[0])+mr],[min(cs[1])-mr, max(cs[1])+mr]])
        ax.set_xlim(lims[0,0], lims[0,1])
        ax.set_ylim(lims[1,0], lims[1,1])
    else:
        ax.set_xlim(lims[0,0], lims[0,1])
        ax.set_ylim(lims[1,0], lims[1,1])
    if multiplot==True:
        
        if colourbar:
            if abs(lims[0,0]-lims[0,1])>abs(lims[1,0]-lims[1,1]):
                loc = 'top'
                ori = 'horizontal'
            else:
                loc = 'right'
                ori = 'vertical'
            cax, cbar_kwds = matplotlib.colorbar.make_axes(ax, location = loc, pad=0.07)#,
            matplotlib.colorbar.ColorbarBase(cax, cmap=cmap,
                                    norm=normcmap,
                                    orientation=ori)
    elif multiplot==False:
        
        if colourbar:
            cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
            ori='vertical'
            matplotlib.colorbar.ColorbarBase(cax, cmap=cmap,
                                    norm=normcmap,
                                    orientation=ori)
    
    return cmap, normcmap, collection


def UpdateDroplets(ax,cmap, normcmap, collection, dyn_var, radii, t):
    ax.set_title('{:.3e}'.format(t))
    clr=[]
    # update the circle colour =
    for ddx, d in enumerate(dyn_var):
        eclr=cmap(normcmap(d)) 
        clr.append(eclr)   
    collection.set_facecolor(clr)
    del(clr)
    plt.pause(0.001)
    
def get_cmap(n, name='prism'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)
    
def ReportResults(filename, RunTimeInputs, cmap_name):
    """Plots graphics from the evaportion data."""
    plt.close('all')
    fig_V = plt.figure()
    ax_V = fig_V.add_subplot(1, 1, 1)
    fig_idx = plt.figure()
    ax_idx = fig_idx.add_subplot(1, 1, 1)
    fig_dVdt = plt.figure()
    ax_dVdt = fig_dVdt.add_subplot(1, 1, 1)
    fig_dt, ax_dt = plt.subplots()
    #fig_tevap = plt.figure()
    #ax_tevap = fig_tevap.add_subplot(1, 1, 1)
    
    xc = np.mean(RunTimeInputs["xcentres"])
    yc = np.mean(RunTimeInputs["ycentres"])
    ND = len(RunTimeInputs["xcentres"])
    
    Results = iom.load_datasets_h5py(filename, ["Time", "Volume", "Theta", "Radius", "dVdt"])
    print("volume:", Results['Volume'])
    print("Time:", Results['Time'])
    t_evap = np.empty([ND])
    for pdx in range(ND):
        name = "D"+str(pdx)
        ax_V.plot(Results['Time']/np.max(Results['Time']), Results['Volume'][:,pdx]*1e6, label=name, marker="o")
        elem=list(np.nonzero(Results['Volume'][:,pdx]))[0][-1] # finds element where V is first zero.
        t_evap[pdx]=Results['Time'][elem]
        if RunTimeInputs['mode']=="CCR":
            ax_idx.plot(Results['Time']/np.max(Results['Time']), Results['Theta'][:,pdx], label=name, marker="o")
        else:
            ax_idx.plot(Results['Time']/np.max(Results['Time']), Results['Radius'][:,pdx], label=name, marker="o")
            
        ax_dVdt.plot(Results['Time']/np.max(Results['Time']), Results['dVdt'][:,pdx]*1e6, label=name)
        s_dt = ax_dt.scatter(RunTimeInputs['xcentres'],RunTimeInputs['ycentres'],\
        c=pm.normalise(t_evap), cmap=cmap_name, vmin=0, vmax=1)
    
    fig_dt.colorbar(s_dt, ax=ax_dt,  orientation='horizontal')
    
    ax_V.set_ylabel(r"$V (\mu L)$")
    ax_V.set_xlabel(r"$t/\tau_{max}$")
    if RunTimeInputs['DNum']<15:
        ax_V.legend()
        ax_idx.legend()
        ax_dVdt.legend()
        
    ax_idx.set_xlabel(r"$t/\tau_{max}$")
    if RunTimeInputs['mode']=="CCR":
        ax_idx.set_ylabel("\u03B8 (\u00b0)")
    else:
        ax_idx.set_ylabel(r"$a$ (mm)")
    ax_dVdt.set_xlabel(r"$t/\tau_{max}$")
    ax_dVdt.set_ylabel(r"$\frac{dV}{dt} (\mu$"+ r"$Ls^{-1})$")

    fig_V.savefig(os.path.join(RunTimeInputs['Directory'],RunTimeInputs['Filename']+"_V.png"))
    fig_idx.savefig(os.path.join(RunTimeInputs['Directory'],RunTimeInputs['Filename']+"_CA.png"))
    fig_dVdt.savefig(os.path.join(RunTimeInputs['Directory'],RunTimeInputs['Filename']+"_dVdt.png"))
    fig_dt.savefig(os.path.join(RunTimeInputs['Directory'],RunTimeInputs['Filename']+"_drytime_heatmap.png"))

    iom.write_hdf5_directly(t_evap, 't_evap', filename)
    plt.show()
    return

def export_video(data_path, RunTimeInputs, odpi=200, vid_FPS=25, number_of_frames=10, cmap_name='jet'):
    DTM_data = iom.load_datasets_h5py(data_path, ["t_evap", "t_print", "Theta", "Radius", "Time"])
    unique_drying_times = np.unique(DTM_data['t_evap'])
    max_time = np.max(unique_drying_times)+RunTimeInputs['dt']

    times = np.linspace(0,max_time,number_of_frames)
    xs = RunTimeInputs['xcentres']
    ys = RunTimeInputs['ycentres']
    width, height = int(12.80*odpi), int(10.40*odpi)
    out = cv2.VideoWriter(os.path.join(RunTimeInputs['Directory'], RunTimeInputs['Filename']+"video.avi"), cv2.VideoWriter_fourcc(*'MJPG'), vid_FPS, (width,height))
    current_backend = plt.get_backend()
    plt.switch_backend('Agg')
    # Create a figure (no need to show it)
    fig, ax = plt.subplots(constrained_layout=True)  # match size
    printed = DTM_data["t_print"]==0
    centres=list(zip(list(xs[printed]*1e6),list(ys[printed]*1e6)))
    vmax = [0,np.max(RunTimeInputs["CA"])*180/np.pi]
    all_centres=list(zip(list(xs*1e6),list(ys*1e6)))
    cs=list(zip(*all_centres))
    mr=max(RunTimeInputs['Rb'])*1e6
   
    lims= np.array([[min(cs[0])-mr,max(cs[0])+mr],[min(cs[1])-mr,max(cs[1])+mr]])

    cmap1, normcmap1, collection1=CreateDroplets(ax, fig, cmap_name, centres, 
                                                    RunTimeInputs['Rb'][printed]*1e6, RunTimeInputs["CA"][printed]*180/np.pi, vmax[0], vmax[1], True, True, lims=lims)
    
    ax.set_aspect('equal', adjustable='box')
    
    title = fig.suptitle("")
    for tdx, t in enumerate(times):
        
        
        print("writing frame: ", tdx+1, "/"+str(number_of_frames))
        t_i = np.argmin(abs(DTM_data['Time']-t))
        printed = DTM_data["t_print"]<=t_i
        centres=list(zip(list(xs[printed]*1e6),list(ys[printed]*1e6)))
        if RunTimeInputs['mode']=="CCR":
            r0=RunTimeInputs['Rb'][printed]
            theta = DTM_data["Theta"][t_i,:][printed]
            UpdateDroplets(ax, cmap1, normcmap1, collection1, theta*180/np.pi,r0, t)
            title.set_text(f"t = {t:.2f} (s)")

        else:
            ax.clear()
            r0 = DTM_data["Radius"][t_i,:][printed]
            CreateDroplets(ax, fig, cmap_name, centres, r0*1e6, RunTimeInputs["CA"][printed]*180/np.pi, vmax[0], vmax[1], True, False, lims=lims)
            ax.set_aspect('equal', adjustable='box')
            title.set_text(f"t = {t:.2f}")
    
        # Save the figure to a buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', pad_inches=0, dpi=odpi)
        buf.seek(0)

        # Convert buffer to image
        img = Image.open(buf)
        img = img.convert('RGB')  # remove alpha if present
        img = img.resize((width, height))  # ensure size matches
        frame = np.array(img)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        out.write(frame_bgr)

    out.release()
    print("exported")
    plt.switch_backend(current_backend)
    return    

def Compare2Data(tResults, eResults, cmap_name):
    """Comparing drying time results from MDL and MDLT code outputs. 
        Inputs: Dictionaries generated by MDLT and MDL respectively."""

    # plotting axes
    #f_cc, a_cc = plt.subplots(1,1, frameon=True)
    f_dt, ax_dt = plt.subplots(1,1, frameon=True)
    f_lin, ax_lin = plt.subplots(1,1, frameon=True)
    f_2hm, ax_2hm = plt.subplots(1,3)

    # figure  - Drying times against radial distance from centre
    ax_dt.scatter(eResults['rdx']*eResults['s_cal'],pm.normalise(eResults['drying_times']*eResults['t_cal']), \
        color='k', label="Experiment")
    rdl = pm.get_radial_position(tResults["RunTimeInputs"]["xcentres"], tResults["RunTimeInputs"]["ycentres"],\
         np.mean(tResults["RunTimeInputs"]["xcentres"]), np.mean(tResults["RunTimeInputs"]["ycentres"]))*1000
    ax_dt.scatter(rdl,pm.normalise(tResults['t_evap']), color='b', label="Theory")
    
    c=pm.normalise(eResults['drying_times']*eResults['t_cal'])
    centres=list(zip(list(eResults['X']*eResults['s_cal']),list(eResults['Y']*eResults['s_cal'])))
    cmap1, normcmap1, collection1=CreateDroplets(ax_2hm[0], f_2hm, cmap_name, centres, 
                                                    tResults['RunTimeInputs']['Rb']*1000, c, 0, 1, True)
    centres=list(zip(list(tResults['RunTimeInputs']['xcentres']*1000),list(tResults['RunTimeInputs']['ycentres']*1000)))
    c = pm.normalise(tResults['t_evap'])
    cmap1, normcmap1, collection1=CreateDroplets(ax_2hm[1], f_2hm, cmap_name, centres, 
                                                    tResults['RunTimeInputs']['Rb']*1000, c, 0, 1, True)

    diff = pm.normalise(tResults['t_evap'])-pm.normalise(eResults['drying_times'])
    lim = np.max([abs(np.min(diff)), abs(np.max(diff))])
    cmap1, normcmap1, collection1=CreateDroplets(ax_2hm[2], f_2hm, 'RdBu_r', centres, 
                                                    tResults['RunTimeInputs']['Rb']*1000, diff, -lim, lim, True)
    
    # figure - linearising
    xline = np.sqrt(np.max(eResults['rdx']*eResults['s_cal'])**2-(eResults['rdx']*eResults['s_cal'])**2)
    yline = pm.normalise(eResults['drying_times']*eResults['t_cal'])
    ylint = pm.normalise(tResults['t_evap'])
    ax_lin.scatter(yline,ylint, color='k')
    ll = np.min([np.min(yline),np.min(ylint)])
    ax_lin.plot([ll,1],[ll,1])

    print(np.max((yline-ylint)/yline))
    print(np.min((yline-ylint)/yline))
    print(np.mean((yline-ylint)/yline))

    # labeling + formatting
    ax_lin.set_ylabel(r"$\tau_{th}$")
    ax_lin.set_xlabel(r"$\tau_{exp}$")

    ax_dt.set_ylabel(r"$\hat{\tau}$")
    ax_dt.set_xlabel(r"$r_c$")
    ax_dt.legend()

    ax_2hm[0].set_title("Experiment")
    ax_2hm[1].set_title("Theory")
    ax_2hm[2].set_title("Difference")
    ax_2hm[0].set_aspect('equal')
    ax_2hm[1].set_aspect('equal')
    ax_2hm[2].set_aspect('equal')
    
    # Saving
    f_2hm.savefig(os.path.join(tResults['RunTimeInputs']['Directory'], tResults['RunTimeInputs']['Filename']+"scatter.png"))
    f_lin.savefig(os.path.join(tResults['RunTimeInputs']['Directory'], tResults['RunTimeInputs']['Filename']+"te_tau.png"))
    f_dt.savefig(os.path.join(tResults['RunTimeInputs']['Directory'], tResults['RunTimeInputs']['Filename']+"taur.png"))
    # f_2hm.savefig(os.path.join(tResults['RunTimeInputs']['Directory'], tResults['RunTimeInputs']['Filename']+"scatter.svg"))
    # f_lin.savefig(os.path.join(tResults['RunTimeInputs']['Directory'], tResults['RunTimeInputs']['Filename']+"te_tau.svg"))
    # f_dt.savefig(os.path.join(tResults['RunTimeInputs']['Directory'], tResults['RunTimeInputs']['Filename']+"taur.svg"))
    
    plt.show()
    return
