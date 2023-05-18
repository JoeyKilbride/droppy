import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes._axes import _log as matplotlib_axes_logger
from copy import deepcopy
import math
import droppy as dpy
import timeit
import time
import sys
matplotlib_axes_logger.setLevel('ERROR')
sys.path.insert(0, r'/Volumes/ERD160_projects$/aaaa_Joey/Scripts/')
import Visualisation as vis

# =============================================================================
# import matplotlib
# from matplotlib.animation import FFMpegWriter
# import os
# import cProfile
# import cython
# import cdroppy
# import matplotlib.colors as colors
# from matplotlib.collections import PatchCollection
# from matplotlib.animation import FuncAnimation
# import itertools
# import array
# from PIL import ImageColor
# import pickle
# 
# =============================================================================


def WrayEvaporate(RunTimeInputs):
    """Runs Wray et al. Model iteratively based on input parameters."""

    #plt.close('all')
    print("_____________________________________________")
    print("Starting Parameters:")
    print("_____________________________________________")
      
    # Ready the troops
    xcentres =RunTimeInputs['xcentres']
    ycentres =RunTimeInputs['ycentres']
    r0       =RunTimeInputs['Rb']
    Vi       =RunTimeInputs['Vi']
    t        =RunTimeInputs['t']

    gone     = np.ones(RunTimeInputs['DNum'])==1
    alive    = np.ones(RunTimeInputs['DNum'])==1
    V        = np.ones(RunTimeInputs['DNum'])
    V_t      = np.empty((0,RunTimeInputs['DNum']), float)
    dVdt_t   = np.empty((0,RunTimeInputs['DNum']), float)
    t_i      = np.empty((0,1), float)
    theta_t  = np.empty((0,RunTimeInputs['DNum']), float)
    dVdt     = np.zeros(RunTimeInputs['DNum'], float)
    Calc_Time= np.empty(0,float)
    transient_length = RunTimeInputs['Transient_Length'] 

    gone_record=gone
    step_counter=0
    
    ## Setup plotting 
    #metadata = dict(title='Movie Test', artist='Matplotlib',
    #            comment='Movie support!')
    #writer   = FFMpegWriter(fps=30, metadata=metadata)
    #colors   = iter(matplotlib.cm.rainbow(np.linspace(0, 1, RunTimeInputs['DNum'])))
    
    fig, (ax1, ax2) = plt.subplots(1, 2)
    plt.title("Droplet Layout")
    # ax1.set_aspect(1)
    # ax2.set_aspect(1)
    # # xmax=max(xcentres)+max(r0)
    # xmin=min(xcentres)-max(r0)
    # ymax=max(ycentres)+max(r0)      s
    # ymin=min(ycentres)-max(r0)
    # ax1.set_xlim(xmin, xmax)
    # ax1.set_ylim(ymin, ymax)
    # ax2.set_xlim(xmin, xmax)
    # ax2.set_ylim(ymin, ymax)

    centres=list(zip(list(xcentres),list(ycentres)))
    dVdt_iso    = dpy.getIsolated(RunTimeInputs['Ambient_T'], RunTimeInputs['Ambient_RH'], r0, 0, RunTimeInputs['rho_liquid']) # Using Hu & Larson 2002 eqn. 19
    cmap1, normcmap1, collection1=dpy.CreateDroplets(ax1, fig, 'gnuplot2_r', centres,r0, RunTimeInputs['CA']*(180/np.pi),0,90, True)
    cmap2, normcmap2, collection2=dpy.CreateDroplets(ax2, fig, 'seismic_r', centres,r0, np.zeros(RunTimeInputs['DNum']),max(-dVdt_iso)*-100,max(-dVdt_iso)*100, True)



    #cmap, normcmap, collection=CreateDroplets(ax, fig, 'gnuplot2_r', xcentres, ycentres,r0, RunTimeInputs['CA']*(180/pi))
    # UpdateDroplets(ax, cmap, normcmap, collection, RunTimeInputs['CA']*(180/pi), RunTimeInputs['t'])
    # plt.pause(0.1)
    residual=0
    ZERO=min(Vi)/1000 # IS THIS CORRECT???
    print("\tNumber of droplets = ", RunTimeInputs['DNum'])
    #print("\tInitial Base Radius (m) = ", r0)
    #print("\tInitial Volume (\u03BC"+ "L)= ", Vi)
    #print("\tInitial Contact Angle (\u00B0) = ", RunTimeInputs['CA']*(180/np.pi))
    print("_____________________________________________")
    dVdt_iso=dpy.getIsolated(RunTimeInputs['Ambient_T'], RunTimeInputs['Ambient_RH'], RunTimeInputs['Rb'], 0, RunTimeInputs['rho_liquid']) # Using Hu & Larson 2002 eqn. 19
    #with writer.saving(fig, os.path.join(RunTimeInputs['directory'],RunTimeInputs['filename']+".avi"), 100):
    transient_times = np.zeros(RunTimeInputs['DNum'])
    transient_droplets=np.zeros(RunTimeInputs['DNum'], dtype=bool) # none initially transient
    #print("transient_droplets", transient_droplets)
    dVdt_last =dpy.WrayFabricant(xcentres[alive], ycentres[alive], r0[alive], dVdt_iso[alive])# F0 in m/s (was 3.15e-07) # Place holder for future declaration
    while len(V[alive])>ZERO: # ?Can i make the Vi[alive] and remove the variable V?
        
        tic = time.perf_counter()
        dVdt_new=dpy.WrayFabricant(xcentres[alive], ycentres[alive], r0[alive], dVdt_iso[alive])# F0 in m/s (was 3.15e-07)
        toc = time.perf_counter()
        Calc_Time = np.append(Calc_Time,toc-tic)
        dVdt[alive]=deepcopy(dVdt_new) # update new evaporation rates for living droplets
        dVdt[transient_droplets] = deepcopy(dVdt_last[transient_droplets])
        dVdt = np.where(Vi>=ZERO,dVdt,0) # dead droplets evaporation rates set to 0
        del_t=Vi[alive]/-dVdt[alive]
        fastest=np.where(del_t[del_t>0]==np.min(del_t[del_t>0]))
        dt_fastest=del_t[del_t>0][fastest][0] # first element - all should be the same (coof_var should reveal if not)
        coof_var=np.std(del_t[del_t>0][fastest])/np.mean(del_t[del_t>0][fastest])
        print("Coefficient of variance (should be low) = ",coof_var)
        
        if np.any(transient_times<0):
            dt_transient=np.min(-transient_times[alive][transient_droplets[alive]]) # time till next transient period ends
            i_del_t = np.min([dt_transient,dt_fastest]) # step smallest time step
            t_transient=True
        else:
            i_del_t = dt_fastest
            t_transient=False
        #print("time_step_selected: ",i_del_t)
        while not(np.any(Vi[alive] <= ZERO)):
            #print("Transient_Droplets", transient_droplets)
            #dVdt[transient_droplets] = dVdt_last[transient_droplets]
            t_i = np.vstack([t_i, t])
            V_t=np.vstack([V_t, Vi])# add new volumes to array
            #print("V[235]=",Vi[235])
            theta=dpy.GetCAfromV(Vi/1000, r0, ZERO)
            theta_t=np.vstack([theta_t, theta*180/np.pi])
            dVdt_t=np.vstack([dVdt_t, dVdt])# add new evap rates                
            Vprev = deepcopy(Vi)
            t=math.fsum([t,i_del_t]) # increment time
            residual= residual+np.sum(Vi[np.where(Vi<ZERO)]) # sum overshoots
            step_counter=step_counter+1 # 
            print("Step ",str(step_counter)+", \u0394"+"t = (+",i_del_t,")s")

            Vi = Vprev+(dVdt*i_del_t) # Reduce volume 
            #print("Current Volume"+"(\u03BC"+ "L)=",Vi) 
            #writer.grab_frame()
            dpy.UpdateDroplets(ax1, cmap1, normcmap1, collection1, theta*180/np.pi, t)
            dpy.UpdateDroplets(ax2, cmap2, normcmap2, collection2, dVdt, t)
            #writer.grab_frame()
            #plt.pause(0.01)
            
            # delaying evaporation rate change
            #transient_times[alive]=math.fsum([transient_times[alive],i_del_t])
        
            transient_times[transient_droplets]=np.array([math.fsum([x,i_del_t]) for x in transient_times[transient_droplets]])
            transient_droplets = transient_times<0 # update transient droplets
            #print("updated transient times: ",transient_times)
            break
        
            

            
                
        print("_____________________________________________")
        print("Timestep Complete ...")
        gone  = Vi<ZERO
        alive = Vi>=ZERO
        N_alive = len(Vi[alive])
        dVdt_last=deepcopy(dVdt)
        transient_droplets[gone]=np.zeros(RunTimeInputs['DNum']-N_alive, dtype=bool)# gone droplet are never transient
        if not(t_transient) and transient_length>0:
            transient_droplets[alive]=np.ones(N_alive, dtype=bool)
            transient_times[alive] -= np.ones(N_alive)*transient_length # sum multiple transient periods
        # if not(t_transient): # if droplets have evaporated then add transient
        #     transient_times[gone] = np.zeros(RunTimeInputs['DNum']-N_alive)
        #     transient_times[alive] -= np.ones(N_alive)*transient_length # sum multiple transient periods
        #     print("min: ",min(transient_times), "max: ", max(transient_times))
        #     transient_droplets = transient_times<0 # update transient droplets
        # else:
        #     transient_droples =  np.zeros(N_alive, dtype=bool) # must all now be non transient
        print("\tNo. of alive droplets left: ", N_alive)
        print("_____________________________________________")
        Vi = np.where(Vi>=ZERO,Vi,0)
        gone_record=np.vstack([gone_record, gone])
        #print("Contact Angle (\u00B0)= ", theta*180/np.pi)
        
    V_t=np.vstack([V_t, np.zeros(len(Vi))])# add zero volumes to array
    dVdt_t=np.vstack([dVdt_t, dVdt])# add new volumes to array
    t_i = np.vstack([t_i, t]) # add final times to array
    theta=dpy.GetCAfromV(Vi/1000, r0, ZERO)
    theta_t=np.vstack([theta_t, theta*180/np.pi])
    
    print("_____________________________________________")
    print("Residual Volume error (should=0)= ", residual)
    print("Droplets took ",t/60, " mins to evaporate")
    print("Completed in ", step_counter, "steps")
    print("_____________________________________________")
    
    WrayResults={}
    WrayResults["Time"]=t_i
    WrayResults["Volume"]=V_t
    WrayResults["Theta"]= theta_t
    WrayResults["dVdt"]= dVdt_t
    WrayResults["RunTimeInputs"]=RunTimeInputs
    WrayResults["Calc_Time"]=Calc_Time    
    return WrayResults
#
#
#
#
#
#
#
#
#
#
def MasoudEvaporate(RunTimeInputs):
    """main function."""
    
    print("_____________________________________________")
    print("Starting Parameters:")
    print("_____________________________________________")
      
    # Ready the troops
    xcentres = RunTimeInputs['xcentres']
    ycentres = RunTimeInputs['ycentres']
    r0       = RunTimeInputs['Rb']
    theta    = RunTimeInputs['CA']
    t        = RunTimeInputs['t']
    Vi       = RunTimeInputs['Vi']
    dt       = RunTimeInputs['dt'] 
    RH       = RunTimeInputs['Ambient_RH']
    gone        = np.ones(RunTimeInputs['DNum'])==1
    alive       = np.ones(RunTimeInputs['DNum'])==1
    V           = np.ones(RunTimeInputs['DNum'])
    V_t         = np.empty((0,RunTimeInputs['DNum']), float)
    r0_t        = np.empty((0,RunTimeInputs['DNum']), float)
    dVdt_t      = np.empty((0,RunTimeInputs['DNum']), float)
    t_i         = np.empty((0,1), float)
    RH_t        = np.empty((0,1), float)
    theta_t     = np.empty((0,RunTimeInputs['DNum']), float)
    dVdt        = np.zeros(RunTimeInputs['DNum'], float)
    transient_length = RunTimeInputs['Transient_Length'] 
    gone_record = gone
    alive_prev  = deepcopy(alive)

    #h0       =RunTimeInputs['h0']
    
    ## Setup plotting 
    #metadata = dict(title='Movie Test', artist='Matplotlib',
    #            comment='Movie support!')
    #writer = FFMpegWriter(fps=30, metadata=metadata)
    #colors = iter(matplotlib.cm.rainbow(np.linspace(0, 1, RunTimeInputs['DNum'])))
    fig, (ax1, ax2) = plt.subplots(1, 2)
    plt.title("Droplet Layout")
    # ax1.set_aspect(1)
    # ax2.set_aspect(1)
    # xmax=max(xcentres)+max(r0)
    # xmin=min(xcentres)-max(r0)
    # ymax=max(ycentres)+max(r0) 
    # ymin=min(ycentres)-max(r0)
    # ax1.set_xlim(xmin, xmax)
    # ax1.set_ylim(ymin, ymax)
    # ax2.set_xlim(xmin, xmax)
    # ax2.set_ylim(ymin, ymax)
    cmtype1='gnuplot2_r'
    cmtype2='hot'
    
    #CreateDroplets(ax, fig, cmaptype, centres, r0, C, vmin, vmax, multiplot)
    centres=list(zip(list(xcentres),list(ycentres)))
    dVdt_iso    = dpy.getIsolated(RunTimeInputs['Ambient_T'], RunTimeInputs['Ambient_RH'], r0, RunTimeInputs['CA'], RunTimeInputs['rho_liquid']) # Using Hu & Larson 2002 eqn. 19
    vmax1 = [0,90]
    vmax2 = [max(dVdt_iso*1000)/4,0]
    cmap1, normcmap1, collection1=vis.CreateDroplets(ax1, fig, cmtype1, centres, r0, RunTimeInputs['CA']*(180/np.pi),vmax1[0],vmax1[1], True)
    cmap2, normcmap2, collection2=vis.CreateDroplets(ax2, fig, cmtype2, centres, r0, np.zeros(RunTimeInputs['DNum']),vmax2[0],vmax2[1], True)#'RdYlGn'
    
    residual=0
    ZERO=min(Vi)/10000 # IS THIS CORRECT???
    print("\tNumber of droplets = ", RunTimeInputs['DNum'])
    #print("\tr0 = ", r0)
    #print("\tInitial Volume (\u03BC"+ "L)= ", Vi)
    #print("\tInitial Contact Angle (\u00B0) = ", RunTimeInputs['CA']*(180/np.pi))
    print("_____________________________________________")
    #with writer.saving(fig, "Evaporation_N="+str(RunTimeInputs['DNum'])+".avi", 100):
    loop_start=timeit.default_timer()    
    
    transient_times = np.zeros(RunTimeInputs['DNum'])
    transient_droplets=np.zeros(RunTimeInputs['DNum'], dtype=bool) # none initially transient
    dVdt_transient=np.zeros(RunTimeInputs['DNum'])
    while len(V[alive])>0: # ?Can i make the Vi[alive] and remove the variable V?
        
        print("___________________Remaining Evaporating____________________")
        
        while not(any(Vi[alive] <= ZERO)):
            if (RunTimeInputs['mode'] == "CCR"):
                theta = dpy.GetCAfromV(Vi/1000, r0, ZERO)  
                #print("Contact Angle (\u00B0)= ", theta*180/np.pi)
            elif (RunTimeInputs['mode'] == "CCA"):
                r0 = dpy.GetBase(theta, Vi/1000)
                #print("Base radius (m)= ", r0)
                         

            Vprev       = deepcopy(Vi)
            #tic = time.perf_counter()

            dVdt_iso    = dpy.getIsolated(RunTimeInputs['Ambient_T'], RH, r0, theta, RunTimeInputs['rho_liquid']) # Using Hu & Larson 2002 eqn. 19
            dVdt_new    = dpy.Masoud(xcentres[alive], ycentres[alive], r0[alive], dVdt_iso[alive], theta[alive])
            
            #toc = time.perf_counter()
            #Calc_Time[step_counter]=toc-tic
            #dVdt_WF     = WrayFabricant(xcentres[alive], ycentres[alive], r0[alive], dVdt_iso[alive])# F0 in m/s (was 3.15e-07)
            dVdt[alive] = deepcopy(dVdt_new) # update new evaporation rates for living droplets 
            dVdt        = np.where(Vi!=ZERO,dVdt,0) # dead droplets evaporation rates set to 0
            
            if np.sum(transient_droplets)>0: # if any transient droplets
                dVdt_transient[alive_prev]    = dpy.Masoud(xcentres[alive_prev], 
                                               ycentres[alive_prev], r0[alive_prev], 
                                               dVdt_iso[alive_prev], theta[alive_prev])
                dVdt[transient_droplets] = deepcopy(dVdt_transient[transient_droplets])
            t_i     = np.vstack([t_i, t]) # record time steps
            V_t     = np.vstack([V_t, Vi]) # add new volumes to array
            r0_t    = np.vstack([r0_t, r0])
            theta_t = np.vstack([theta_t, theta*180/np.pi])
            
            dVdt_t  = np.vstack([dVdt_t, dVdt])# add new volumes to array
            t        = math.fsum([t,dt])
            Vi       = Vprev+(dVdt*dt)
            residual = residual+sum(Vi[np.where(Vi<ZERO)])
            if RunTimeInputs['box_volume']!=np.inf:
                print("RH = ","{:.2f}".format(RH*100),"%")
                RH_t    = np.vstack([RH_t, RH]) 
                RH          = dpy.dynamic_humidity(RunTimeInputs['box_volume'],RunTimeInputs['molar_mass'],
                                        RunTimeInputs['Antoine_coeffs'][0],
                                        RunTimeInputs['Antoine_coeffs'][1],
                                        RunTimeInputs['Antoine_coeffs'][2],
                                        RunTimeInputs['Ambient_T'] +273.15, 
                                        RunTimeInputs['rho_liquid'], np.sum(-1*(dVdt*dt))) + RH_t[-1][0]

            else:
                RH = RunTimeInputs['Ambient_RH']
            # writer.grab_frame()
            if RunTimeInputs['mode']=="CCR":
                dpy.UpdateDroplets(ax1, cmap1, normcmap1, collection1, theta*180/np.pi,r0, t)
                dpy.UpdateDroplets(ax2, cmap2, normcmap2, collection2, dVdt,r0, t)
            else:
                ax1.clear()
                ax2.clear()
                vis.CreateDroplets(ax1, fig, cmtype1, centres, r0, theta*180/np.pi, vmax1[0], vmax1[1], None)
                vis.CreateDroplets(ax2, fig, cmtype2, centres, r0, dVdt, vmax2[0], vmax2[1], None)
                plt.pause(0.001)
            #writer.grab_frame()
            
            #print("Current Volume"+"(\u03BC"+ "L)=",Vi/1e-6)
            #print("Evaporation Rate"+"(\u03BC"+ "L/s)=",dVdt/1e-6)
            
            transient_times[transient_droplets]=np.array([math.fsum([x,dt]) for x in transient_times[transient_droplets]])
            transient_droplets = transient_times<0 # update transient droplets
            print("| "+str(t)+" ",end="", flush=True)
        
        gone        = Vi<ZERO
        alive_prev=deepcopy(alive)
        alive       = Vi>=ZERO
        N_alive = len(Vi[alive])
        gone_record = np.vstack([gone_record, gone])
        
        transient_droplets[gone]=np.zeros(RunTimeInputs['DNum']-N_alive, dtype=bool)# gone droplet are never transient
        #transient_droplets[alive]=np.ones(N_alive, dtype=bool)
        transient_times[alive] -= np.ones(N_alive)*transient_length
        transient_droplets = transient_times<0
        Vi   = np.where(Vi>=ZERO,Vi,0) # sets negative values to zero volume
        dVdt = np.where(Vi>=ZERO,dVdt,0)

        print("_____________________________________________")
        print("Droplets have evaporated, updating matrix ...")
        print("\tNo. of alive droplets left: ", len(Vi[alive]))
        print("_____________________________________________")
    
    loop_end=timeit.default_timer()
    print("Loop ran for: ",loop_end-loop_start)
    V_t=np.vstack([V_t, np.zeros(len(Vi))])# add zero volumes to array
    t_i = np.vstack([t_i, t]) # add final times to array
    if (RunTimeInputs['mode'] == "CCR"):
        theta = dpy.GetCAfromV(Vi/1000, r0, ZERO) 
        theta_t= np.vstack([theta_t, theta*180/np.pi])
        # update final plot
        dpy.UpdateDroplets(ax1, cmap1, normcmap1, collection1, theta*180/np.pi,r0, t)
        dpy.UpdateDroplets(ax2, cmap2, normcmap2, collection2, dVdt,r0, t)
    elif (RunTimeInputs['mode'] == "CCA"):
        r0 = dpy.GetBase(theta, Vi/1000)
        r0_t    = np.vstack([r0_t, r0])
        # update final plot
        ax1.clear()
        ax2.clear()
        vis.CreateDroplets(ax1, fig, cmtype1, centres, r0, theta*180/np.pi, vmax1[0], vmax1[1], None)
        vis.CreateDroplets(ax2, fig, cmtype2, centres, r0, dVdt, vmax2[0], vmax2[1], None)
        plt.pause(0.001)

    dVdt_t  = np.vstack([dVdt_t, dVdt])# add new volumes to array
    print("_____________________________________________")
    print("Residual Volume error = ", residual)
    print("Droplets took ",t/60, " mins to evaporate")
    print("_____________________________________________")
    print(type(V_t))
    MasoudResults={}
    MasoudResults["Time"]=t_i
    MasoudResults["Volume"]=V_t
    MasoudResults["dVdt"]=dVdt_t
    if RunTimeInputs['box_volume']!=np.inf:
        MasoudResults["Ambient_RH"]=RH_t
    else:
        MasoudResults["Ambient_RH"]=RunTimeInputs["Ambient_RH"]
    if (RunTimeInputs['mode'] == "CCR"):
        MasoudResults["Theta"] = theta_t
    elif (RunTimeInputs['mode'] == "CCA"):
        MasoudResults['Radius'] = r0_t
    
    MasoudResults["RunTimeInputs"]=RunTimeInputs
    #WrayResults["Calc_Time"]=Calc_Time

    return MasoudResults


    