import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.axes._axes import _log as matplotlib_axes_logger
from copy import deepcopy
import math
import physics_methods as pm
import io_methods as iom
import visualisation_methods as vm
import timeit
import time
import h5py
matplotlib_axes_logger.setLevel('ERROR')


def Iterate(RunTimeInputs, output_target, plot=False):
    """main function."""
    os.environ["OPENBLAS_NUM_THREADS"] = "4"

    print("_____________________________________________")
    print("Starting Parameters:")
    print("_____________________________________________")

    
    buffer_size = 10 # make this configurable later

    # Ready the troops
    xcentres={}
    xcentres[0] = RunTimeInputs['xcentres']
    ycentres={}
    ycentres[0] = RunTimeInputs['ycentres']
    r0       = RunTimeInputs['Rb']
    theta    = RunTimeInputs['CA']
    t        = 0
    Vi       = RunTimeInputs['Vi']
    dt       = RunTimeInputs['dt'] 
    RH       = RunTimeInputs['Ambient_RHs'][0]
    N = RunTimeInputs['DNum']
    nmols = RunTimeInputs['n_mols']
    gone        = np.ones(N)==1
    alive       = np.ones(N)==1
    V           = np.ones(N)
    connection_history = {}    
    V_t         = np.empty((0,N), float)
    r0_t        = np.empty((0,N), float)
    dVdt_t      = np.empty((0,N), float)
    t_i         = np.empty((0,1), float)
    RH_t        = np.empty((0,1), float)
    theta_t     = np.empty((0,N), float)
    dVdt        = np.zeros(N, float)
    
    xc = xcentres[0]
    yc = ycentres[0]
    
    transient_length = RunTimeInputs['Transient_Length'] 
    gone_record = gone

    if RunTimeInputs['p_rate']==0:
        t_print = np.zeros(RunTimeInputs['DNum'])
    else:
        t_print = np.linspace(0,RunTimeInputs['DNum']/RunTimeInputs['p_rate'],RunTimeInputs['DNum'])

    has_V = Vi>0
    printed = t_print==t # should be 1 droplet

    alive = np.logical_and(printed, has_V)

    alive_prev  = deepcopy(alive)


    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        plt.title("Droplet Layout")

        cmtype1='gnuplot2_r'
        cmtype2='Spectral'
        
    centres=list(zip(list(xc),list(yc)))
    
    csat = pm.ideal_gas_law(pm.Psat(*RunTimeInputs['Antoine_coeffs'][0], RunTimeInputs['Ambient_T']), RunTimeInputs['Ambient_T'],  RunTimeInputs['molar_masses'][0])
    if RunTimeInputs['model'] == "Wray":
        dVdt_iso    = pm.getIsolated(csat ,RH, r0, 0, RunTimeInputs['rho_liquid'], 
                                        RunTimeInputs['D'], RunTimeInputs['molar_masses'][0], 
                                        RunTimeInputs['surface_tension'], RunTimeInputs['Ambient_T'],
                                        RunTimeInputs['n_mols'], RunTimeInputs['i']) # Using Hu & Larson 2002 eqn. 19
    if RunTimeInputs['model'] == "Masoud":
        dVdt_iso    = pm.getIsolated(csat ,RH, r0, theta, RunTimeInputs['rho_liquid'], 
                                                RunTimeInputs['D'], RunTimeInputs['molar_masses'][0], 
                                                RunTimeInputs['surface_tension'], RunTimeInputs['Ambient_T']
                                                ,RunTimeInputs['n_mols'], RunTimeInputs['i']) # Using Hu & Larson 2002 eqn. 19
    rand_evap = np.random.normal(0, RunTimeInputs['rand'], len(dVdt_iso))/100
    dVdt_iso = dVdt_iso+(dVdt_iso*rand_evap)
    if plot:
        vmax1 = [0,np.max(theta)*180/np.pi]

        dVdt_new = pm.Masoud_fast(xc, yc, r0, dVdt_iso, theta)
        vmax2 = [min(dVdt_new),max(dVdt_new)]
        if vmax2[0]>0: # all condensing
            vmax2[0]=0
        if vmax2[1]<0: # all evaporating
            vmax2[1]=0
        cmap1, normcmap1, collection1=vm.CreateDroplets(ax1, fig, cmtype1, centres, r0, RunTimeInputs['CA']*(180/np.pi),vmax1[0],vmax1[1], True,True)
        cmap2, normcmap2, collection2=vm.CreateDroplets(ax2, fig, cmtype2, centres, r0, np.zeros(RunTimeInputs['DNum']),vmax2[0],vmax2[1], True,True)#'RdYlGn'
    
    residual=0
    ZERO=0.0 

    print("\tNumber of droplets = ", N)
    print("_____________________________________________")

    loop_start=timeit.default_timer()    
    transient_times = np.zeros(N)
    transient_droplets=np.zeros(N, dtype=bool) # none initially transient
    dVdt_transient=np.zeros(N)

    if ((RunTimeInputs['bias_point'] is None) or (RunTimeInputs['bias_grad'] is None)):
        no_bias=True
        b_ang=None
        mb = 0
        bias = np.ones(len(xc))
    else:
        b_ang, mb, bias = pm.add_bias(RunTimeInputs['bias_point'][0]/1000, RunTimeInputs['bias_point'][1]/1000, xcentres[alive], ycentres[alive], RunTimeInputs['bias_grad'])
    
    if RunTimeInputs['model'] == "Wray":
         dVdt_new=pm.WrayFabricant(xc[alive], yc[alive], r0[alive], dVdt_iso[alive])
    
    ############################################
    segment_index = 0
    volume_segment = f"Volume_{segment_index:04d}"
    dVdt_segment = f"dVdt_{segment_index:04d}"
    radius_segment = f"radius_{segment_index:04d}"
    theta_segment = f"theta_{segment_index:04d}"
    with h5py.File(output_target+".h5", "w") as f:
        dset1 = f.create_dataset(
            "Time",
            shape=(0,1),             # another dataset
            maxshape=(None,1),
            dtype="float64",
            chunks=(buffer_size,1),
            compression="gzip"
        )
        Vgrp = f.require_group("volume_segments")
        dset2 = Vgrp.create_dataset(
            volume_segment,
            shape=(0,N),             # another dataset
            maxshape=(None,N),
            dtype="float64",
            chunks=(buffer_size,N),
            compression="gzip"
        )
        dVdtgrp = f.require_group("dvdt_segments")
        dset3 = dVdtgrp.create_dataset(
            dVdt_segment,
            shape=(0,N),             # another dataset
            maxshape=(None,N),
            dtype="float64",
            chunks=(buffer_size,N),
            compression="gzip"
        )

        if RunTimeInputs['box_volume']!=np.inf:
            dset5 = f.create_dataset(
                "Ambient_RHs",
                shape=(0,1),             # another dataset
                maxshape=(None,1),
                dtype="float64",
                chunks=(buffer_size,1),
                compression="gzip"
            )
        print("theta_seg:", theta_segment)
        thetagrp = f.require_group("theta_segments")
        dset6 = thetagrp.create_dataset(
            theta_segment,
            shape=(0,N),             # another dataset
            maxshape=(None,N),
            dtype="float64",
            chunks=(buffer_size,N),
            compression="gzip"
        )     
        radiusgrp = f.require_group("radius_segments")
        dset6 = radiusgrp.create_dataset(
            radius_segment,
            shape=(0,N),             # another dataset
            maxshape=(None,N),
            dtype="float64",
            chunks=(buffer_size,N),
            compression="gzip"
        )

    ############################################
    
    while len(Vi[alive])>0: 
        print("___________________Remaining Evaporating____________________")
        any_evaporated = np.any(Vi[alive] <= ZERO)
        any_unprinted = len(t_print<=t)>len(printed)
        touching, any_touching = pm.TouchingCircles(xc[alive],yc[alive],r0[alive],theta[alive])
        keep_incrementing = not(any([any_evaporated, any_unprinted, any_touching]))
        while keep_incrementing:
            if t>RunTimeInputs['t_terminate']:
                print("_________________")
                print("termination time flagged")
                terminate=True
                print("_________________")
                break
            else:
                terminate=False
            if (RunTimeInputs['mode'] == "CCR"):
                theta = pm.GetCAfromV(Vi/1000, r0, ZERO)  
            elif (RunTimeInputs['mode'] == "CCA"):
                r0 = pm.GetBase(theta, Vi/1000)
            print("average radius: ", np.mean(r0[alive]))
            Vprev       = deepcopy(Vi)

            if RunTimeInputs['model'] == "Masoud":
                

                dVdt_iso    = pm.getIsolated(csat ,RH, r0[alive], theta[alive], RunTimeInputs['rho_liquid'], 
                                            RunTimeInputs['D'], RunTimeInputs['molar_masses'][0], 
                                            RunTimeInputs['surface_tension'], RunTimeInputs['Ambient_T'],
                                            nmols[alive], RunTimeInputs['i']) # Using Hu & Larson 2002 eqn. 19
                
                tic = time.perf_counter()
         
                dVdt_new = pm.Masoud_fast(xc[alive], yc[alive], r0[alive], dVdt_iso, theta[alive])
                
                toc = time.perf_counter()
                print(f"Matrix inversion time: {toc-tic:.3f}s")
                dVdt_new=dVdt_new+(dVdt_new) #*rand_evap[alive]
                dVdt[alive] = deepcopy(dVdt_new) #*bias[alive] # update new evaporation rates for living droplets 
            
            if RunTimeInputs['model'] == 'Wray':
                if (RunTimeInputs['mode'] == "CCA"):
                    dVdt_iso    = pm.getIsolated(csat ,RH, r0[alive], theta[alive], RunTimeInputs['rho_liquid'], 
                                            RunTimeInputs['D'], RunTimeInputs['molar_masses'][0], 
                                            RunTimeInputs['surface_tension'], RunTimeInputs['Ambient_T'],
                                            nmols[alive], RunTimeInputs['i']) # Using Hu & Larson 2002 eqn. 19    
                else:
                    dVdt_iso    = pm.getIsolated(csat ,RH, r0[alive], theta[alive], RunTimeInputs['rho_liquid'], 
                                            RunTimeInputs['D'], RunTimeInputs['molar_masses'][0], 
                                            RunTimeInputs['surface_tension'], RunTimeInputs['Ambient_T'],
                                            nmols[alive], RunTimeInputs['i'])
            
                dVdt_new=pm.WrayFabricant(xc[alive], yc[alive], r0[alive], dVdt_iso)
                dVdt[alive] = deepcopy(dVdt_new*bias[alive]) # update new evaporation rates for living droplets
            dVdt        = np.where(Vi>=ZERO,dVdt,0) # dead droplets evaporation rates set to 0
            
            if np.sum(transient_droplets)>0: # if any transient droplets
                if RunTimeInputs['model'] == "Masoud":
                    dVdt_transient[alive_prev]  = pm.Masoud_fast(xc[alive_prev], 
                                                yc[alive_prev], r0[alive_prev], 
                                                dVdt_iso[alive_prev], theta[alive_prev])
                if RunTimeInputs['model'] == "Wray":
                    dVdt_new=pm.WrayFabricant(xc[alive_prev], 
                                               yc[alive_prev], 
                                               r0[alive_prev], 
                                               dVdt_iso[alive_prev])# F0 in m/s (was 3.15e-07)
                
                dVdt[transient_droplets] = deepcopy(dVdt_transient[transient_droplets])
        
            t_i     = np.vstack([t_i, t]) # record time steps
            V_t     = np.vstack([V_t, Vi]) # add new volumes to array
            
            r0_t    = np.vstack([r0_t, r0])
            theta_t = np.vstack([theta_t, theta*180/np.pi])
            
            dVdt_t  = np.vstack([dVdt_t, dVdt])# add new volumes to array
            t        = math.fsum([t,dt])
            Vi       = Vprev+(dVdt*dt)
            print(f"dV/V: {max((dVdt[alive]*dt)/Vi[alive])*100:.3f}%")
            print(f"Volume remaining: {100*(np.sum(Vi)/np.sum(RunTimeInputs['Vi'])):.5f}%")
            residual = residual+sum(Vi[np.where(Vi<ZERO)])
            if RunTimeInputs['box_volume']!=np.inf:
                print("RH = ","{:.5f}".format(RH*100),"%")
                RH_t    = np.vstack([RH_t, RH]) 
                RH          = pm.dynamic_humidity(RunTimeInputs['box_volume'],
                                                RunTimeInputs['molar_masses'][0],
                                                RunTimeInputs['Antoine_coeffs'][0][0],
                                                RunTimeInputs['Antoine_coeffs'][0][1],
                                                RunTimeInputs['Antoine_coeffs'][0][2],
                                                RunTimeInputs['Ambient_T'] +273.15, 
                                                RunTimeInputs['rho_liquid'], 
                                                np.sum(-1*(dVdt*dt))) + RH_t[-1][0]
            
            if len(t_i) == buffer_size:
                print("writing data - buffer full")
                with h5py.File(output_target+".h5", "a") as f:
                    iom.stream_hdf5_collection(f, t_i, "Time")
                    iom.stream_hdf5_collection(f, V_t, volume_segment, group="volume_segments")
                    dVdt_t = iom.stream_hdf5_collection(f, dVdt_t, dVdt_segment, group="dvdt_segments")
                    if RunTimeInputs['box_volume']!=np.inf:
                        iom.stream_hdf5_collection(f, RH_t,"Ambient_RHs")
                        RH_t        = np.empty((0,1), float)
     
                    iom.stream_hdf5_collection(f, theta_t, theta_segment, group="theta_segments")
                    iom.stream_hdf5_collection(f, r0_t, radius_segment, group="radius_segments")


                V_t         = np.empty((0,N), float)
                r0_t        = np.empty((0,N), float)
                dVdt_t      = np.empty((0,N), float)
                t_i         = np.empty((0,1), float)
                theta_t     = np.empty((0,N), float)
                dVdt        = np.zeros(N, float)

                    
            if RunTimeInputs['mode']=="CCR":
                if plot:
                    vm.UpdateDroplets(ax1, cmap1, normcmap1, collection1, theta*180/np.pi,r0, t)
                    vm.UpdateDroplets(ax2, cmap2, normcmap2, collection2, dVdt,r0, t)
            else:
                if plot:
                    ax1.clear()
                    ax2.clear()
                    vm.CreateDroplets(ax1, fig, cmtype1, centres, r0, theta*180/np.pi, vmax1[0], vmax1[1], None,True)
                    vm.CreateDroplets(ax2, fig, cmtype2, centres, r0, dVdt, vmax2[0], vmax2[1], None, True)
                    plt.pause(0.001)
            
            print(f"| {t:2f} ",end="", flush=True)
           
            any_evaporated = np.any(Vi[alive] <= ZERO)

            any_unprinted = np.sum(t_print<=t)>np.sum(printed)
            touching, any_touching = pm.TouchingCircles(xc[alive],yc[alive],r0[alive],theta[alive])

            keep_incrementing = not(any([any_evaporated, any_unprinted, any_touching]))
            
        if any_touching:
            # save anything remaining in the buffer
            with h5py.File(output_target+".h5", "a") as f:
                t_i=iom.stream_hdf5_collection(f, t_i,"Time")
                V_t=iom.stream_hdf5_collection(f, V_t, volume_segment, group="volume_segments")
                dVdt_t = iom.stream_hdf5_collection(f, dVdt_t, dVdt_segment, group="dvdt_segments")
                if RunTimeInputs['box_volume']!=np.inf:
                    RH_t = iom.stream_hdf5_collection(f, RH_t, "Ambient_RHs")
                print("theta_seg: ", theta_segment)
                theta_t = iom.stream_hdf5_collection(f, theta_t, theta_segment, group ="theta_segments")
                r0_t = iom.stream_hdf5_collection(f, r0_t, radius_segment, group="radius_segments")
            
            # which droplets are touching 
            connected = pm.find_chains(touching)
            connection_history[t] = connected

            # finding elements (droplets) which have been absorbed into other elements
            # convention is that if child droplets 0,1,2 are coelescing then the new parent droplet is at element 0.
            # 1 and 2 are deleted.
            nonzero_mask = connected != 0
            _, first_idx = np.unique(connected[nonzero_mask], return_index=True)
            # map back to original indices
            first_idx = np.flatnonzero(nonzero_mask)[first_idx]
            all_idx = np.flatnonzero(nonzero_mask)
            absorbed_indices = np.setdiff1d(all_idx, first_idx)
            
            xc_old = deepcopy(xc[alive])
            yc_old = deepcopy(yc[alive])
            xc = np.delete(xc[alive], absorbed_indices)
            yc = np.delete(yc[alive], absorbed_indices)
            nmols_old = deepcopy(nmols[alive])
            nmols = np.delete(nmols[alive], absorbed_indices)
            Vi_old = deepcopy(Vi)
            Vi = np.delete(Vi[alive],absorbed_indices)
            for i in np.unique(connected)[1:]:
                args = np.argwhere(connected==i)
                xc_i, yc_i, vc_i  = pm.mass_centre(xc_old[args],yc_old[args],Vi_old[alive][args])
                nmols[args[0]] = np.sum(nmols_old[args]) # add mols in each droplet
                xc[args[0]] = xc_i # new centres for the coelesced droplet
                yc[args[0]] = yc_i
                Vi[args[0]] = vc_i # new volume is total of the coelesced droplets
            if (RunTimeInputs['mode'] == "CCR"):
                theta = np.ones(len(Vi))*np.pi/4 # This is a bodge for future sorting out!!!
                                                 # as CCR is a bit unphysical in a coelescence context.
                r0 = pm.GetBase(theta, Vi/1000)
                # theta = pm.GetCAfromV(xcentres[t]/1000, r0, ZERO)  
            elif (RunTimeInputs['mode'] == "CCA"):
                theta = np.ones(len(Vi))*theta[0]
                r0 = pm.GetBase(theta, Vi/1000)
            
            # save the positions for the next segment (post coelescence)
            xcentres[t]=xc
            ycentres[t]=yc
            centres=list(zip(list(xc),list(yc)))
            N -=len(absorbed_indices)

            # start a new segment (post coelescence)
            segment_index+=1
            volume_segment = f"segment_{segment_index:04d}"
            dVdt_segment = f"segment_{segment_index:04d}"
            radius_segment = f"segment_{segment_index:04d}"
            theta_segment = f"segment_{segment_index:04d}"
            V_t         = np.empty((0,N), float)
            r0_t        = np.empty((0,N), float)
            dVdt_t      = np.empty((0,N), float)
            t_i         = np.empty((0,1), float)
            theta_t     = np.empty((0,N), float)
            dVdt        = np.zeros(N, float)
            
        # gone        = Vi<=ZERO
        alive_prev=deepcopy(alive) # Note that this is only used in transients, 
                                   # you can't have coelescence and transients at the same time.

        t_print=np.delete(t_print,absorbed_indices)
        printed = t_print<=t
        print("PRINTED:", printed)
        has_V   = Vi>ZERO
        alive   = np.logical_and(has_V,printed)
        
        # gone_record = np.vstack([gone_record, gone])
        print(Vi)
        print(dVdt)
        Vi   = np.where(Vi>ZERO,Vi,0) # sets negative values to zero volume
        dVdt = np.where(Vi>ZERO,dVdt,0)

        print("_____________________________________________")
        print("Number of droplets has changed, updating matrix ...")
        print("\tNo. of evaporating droplets left: ", len(Vi[has_V]))
        print("\tNo. of printed droplets: ", len(Vi[printed]))
        print("_____________________________________________")
        if terminate:
            break
    
    loop_end=timeit.default_timer()
    tloop=(loop_end-loop_start)/60 # mins
    print(f"Loop ran for: {tloop:2f} mins")


    # Save end values
    V_t=np.vstack([V_t, np.zeros(len(Vi))])# add zero volumes to array
    t_i = np.vstack([t_i, t]) # add final times to array
    
    if (RunTimeInputs['mode'] == "CCR"):
        theta = pm.GetCAfromV(Vi/1000, r0, ZERO) 
        theta_t= np.vstack([theta_t, theta*180/np.pi])
        # update final plot
        if plot:
            vm.UpdateDroplets(ax1, cmap1, normcmap1, collection1, theta*180/np.pi,r0, t)
            vm.UpdateDroplets(ax2, cmap2, normcmap2, collection2, dVdt,r0, t)
    elif (RunTimeInputs['mode'] == "CCA"):
        r0 = pm.GetBase(theta, Vi/1000)
        r0_t    = np.vstack([r0_t, r0])
        if plot:    
            # update final plot
            ax1.clear()
            ax2.clear()
            vm.CreateDroplets(ax1, fig, cmtype1, centres, r0, theta*180/np.pi, vmax1[0], vmax1[1], None, True)
            vm.CreateDroplets(ax2, fig, cmtype2, centres, r0, dVdt, vmax2[0], vmax2[1], None, True)
            plt.pause(0.001)

    dVdt_t  = np.vstack([dVdt_t, dVdt])# add new volumes to array
    with h5py.File(output_target+".h5", "a") as f:
        t_i=iom.stream_hdf5_collection(f, t_i,"Time")
        V_t=iom.stream_hdf5_collection(f, V_t,"Volume")
        dVdt_t = iom.stream_hdf5_collection(f, dVdt_t,"dVdt")
        if RunTimeInputs['box_volume']!=np.inf:
            RH_t = iom.stream_hdf5_collection(f, RH_t,"Ambient_RHs")
        theta_t = iom.stream_hdf5_collection(f, theta_t, theta_segment, group ="theta_segments")
        r0_t = iom.stream_hdf5_collection(f, r0_t, radius_segment, group ="radius_segments")

    iom.write_hdf5_directly(t_print, 't_print', output_target)
    print("_____________________________________________")
    print("Residual Volume error = ", residual)
    print(f"Droplets took ,{t/60:.2f} mins to evaporate")
    print("_____________________________________________")
    
    # print(type(V_t))
    # Results={}
    # Results["Time"]=t_i
    # Results["Volume"]=V_t
    # Results["dVdt"]=dVdt_t
    # if not(no_bias):
    #     Results["bias_grad"]=mb
    #     Results["bias_angle"]=b_ang
    # Results['t_print']=t_print
    # if RunTimeInputs['box_volume']!=np.inf:
    #     Results["Ambient_RHs"]=RH_t
    # if (RunTimeInputs['mode'] == "CCR"):
    #     Results["Theta"] = theta_t
    # elif (RunTimeInputs['mode'] == "CCA"):
    #     Results['Radius'] = r0_t
    
    # Results["RunTimeInputs"]=RunTimeInputs
    #WrayResults["Calc_Time"]=Calc_Time

    return RunTimeInputs
################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################

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

    dVdt_iso    = pm.getIsolated(RunTimeInputs['Ambient_T'], RunTimeInputs['Ambient_RH'], r0, 0, RunTimeInputs['rho_liquid']) # Using Hu & Larson 2002 eqn. 19
    cmap1, normcmap1, collection1=vm.CreateDroplets(ax1, fig, 'gnuplot2_r', centres,r0, RunTimeInputs['CA']*(180/np.pi),0,90, True)
    cmap2, normcmap2, collection2=vm.CreateDroplets(ax2, fig, 'seismic_r', centres,r0, np.zeros(RunTimeInputs['DNum']),max(-dVdt_iso)*-100,max(-dVdt_iso)*100, True)



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
    dVdt_iso=pm.getIsolated(RunTimeInputs['Ambient_T'], RunTimeInputs['Ambient_RH'], RunTimeInputs['Rb'], 0, RunTimeInputs['rho_liquid']) # Using Hu & Larson 2002 eqn. 19
    #with writer.saving(fig, os.path.join(RunTimeInputs['directory'],RunTimeInputs['filename']+".avi"), 100):
    transient_times = np.zeros(RunTimeInputs['DNum'])
    transient_droplets=np.zeros(RunTimeInputs['DNum'], dtype=bool) # none initially transient
    #print("transient_droplets", transient_droplets)
    dVdt_last =pm.WrayFabricant(xcentres[alive], ycentres[alive], r0[alive], dVdt_iso[alive])# F0 in m/s (was 3.15e-07) # Place holder for future declaration
    while len(V[alive])>ZERO: # ?Can i make the Vi[alive] and remove the variable V?
        
        tic = time.perf_counter()
        dVdt_new=pm.WrayFabricant(xcentres[alive], ycentres[alive], r0[alive], dVdt_iso[alive])# F0 in m/s (was 3.15e-07)
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
            theta=pm.GetCAfromV(Vi/1000, r0, ZERO)
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
            vm.UpdateDroplets(ax1, cmap1, normcmap1, collection1, theta*180/np.pi, t)
            vm.UpdateDroplets(ax2, cmap2, normcmap2, collection2, dVdt, t)
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
    theta=pm.GetCAfromV(Vi/1000, r0, ZERO)
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