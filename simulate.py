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
    print_record={}
    printed = t_print==t # should be at least 1 droplet
    print_record[0]=printed
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
    volume_segment = f"segment_{segment_index:04d}"
    dVdt_segment = f"segment_{segment_index:04d}"
    radius_segment = f"segment_{segment_index:04d}"
    theta_segment = f"segment_{segment_index:04d}"
    with h5py.File(output_target+".h5", "w") as f:
        dset1 = f.create_dataset(
            "Time",
            shape=(0,1),             # another dataset
            maxshape=(None,1),
            dtype="float64",
            chunks=(buffer_size,1),
            compression="gzip"
        )
        Vgrp = f.require_group("Volumes")
        dset2 = Vgrp.create_dataset(
            volume_segment,
            shape=(0,N),             # another dataset
            maxshape=(None,N),
            dtype="float64",
            chunks=(buffer_size,N),
            compression="gzip"
        )
        dVdtgrp = f.require_group("dVdt")
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
        
        thetagrp = f.require_group("Theta")
        dset6 = thetagrp.create_dataset(
            theta_segment,
            shape=(0,N),             # another dataset
            maxshape=(None,N),
            dtype="float64",
            chunks=(buffer_size,N),
            compression="gzip"
        )     
        radiusgrp = f.require_group("Radius")
        dset6 = radiusgrp.create_dataset(
            radius_segment,
            shape=(0,N),             # another dataset
            maxshape=(None,N),
            dtype="float64",
            chunks=(buffer_size,N),
            compression="gzip"
        )

    ############################################
    terminate=False
    while len(Vi[alive])>0: 
        print("___________________Remaining Evaporating____________________")
        any_evaporated = np.any(Vi[alive] <= ZERO)
        any_unprinted = len(t_print<=t)>len(printed)
        touching, any_touching = pm.TouchingCircles(xc[alive],yc[alive],r0[alive],theta[alive])
        keep_incrementing = not(any([any_evaporated, any_unprinted, any_touching]))
        print("keep_incremnenting: ", keep_incrementing)
        print(any_evaporated, any_unprinted, any_touching)
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
                    iom.stream_hdf5_collection(f, V_t, volume_segment, group="Volume")
                    dVdt_t = iom.stream_hdf5_collection(f, dVdt_t, dVdt_segment, group="dVdt")
                    if RunTimeInputs['box_volume']!=np.inf:
                        iom.stream_hdf5_collection(f, RH_t,"Ambient_RHs")
                        RH_t        = np.empty((0,1), float)
     
                    iom.stream_hdf5_collection(f, theta_t, theta_segment, group="Theta")
                    iom.stream_hdf5_collection(f, r0_t, radius_segment, group="Radius")


                V_t         = np.empty((0,N), float)
                r0_t        = np.empty((0,N), float)
                dVdt_t      = np.empty((0,N), float)
                t_i         = np.empty((0,1), float)
                theta_t     = np.empty((0,N), float)
                

                    
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

            dVdt        = np.zeros(N, float)
            print(f"| {t:2f} ",end="", flush=True)
           
            any_evaporated = np.any(Vi[alive] <= ZERO)

            any_unprinted = np.sum(t_print<=t)>np.sum(printed)
            touching, any_touching = pm.TouchingCircles(xc[alive],yc[alive],r0[alive],theta[alive])

            keep_incrementing = not(any([any_evaporated, any_unprinted, any_touching]))
            
        if any_touching:
            print("===================================")
            print("===================================")
            print("===================================")
            print("===================================")
            print("handling coelescence____________________")
            # save anything remaining in the buffer
            with h5py.File(output_target+".h5", "a") as f:
                t_i=iom.stream_hdf5_collection(f, t_i,"Time")
                V_t=iom.stream_hdf5_collection(f, V_t, volume_segment, group="Volume")
                dVdt_t = iom.stream_hdf5_collection(f, dVdt_t, dVdt_segment, group="dVdt")
                if RunTimeInputs['box_volume']!=np.inf:
                    RH_t = iom.stream_hdf5_collection(f, RH_t, "Ambient_RHs")
                
                theta_t = iom.stream_hdf5_collection(f, theta_t, theta_segment, group ="Theta")
                r0_t = iom.stream_hdf5_collection(f, r0_t, radius_segment, group="Radius")
            
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
            for idx, i in enumerate(np.unique(connected)[1:]):
                args = np.argwhere(connected==i)
                if len(first_idx[first_idx<=i])==0:
                    ii=0
                else:
                    ii=max(first_idx[first_idx<=i]) # indices below current 
                coaelescing = connected[:int(ii)]
                values, counts = np.unique(coaelescing[coaelescing>0], return_counts=True)
                index_map = np.sum(counts-1)
                xc_i, yc_i, vc_i  = pm.mass_centre(xc_old[args],yc_old[args],Vi_old[alive][args])
                nmols[args[0]-index_map] = np.sum(nmols_old[args]) # add mols in each droplet
                xc[args[0]-index_map] = xc_i # new centres for the coelesced droplet
                yc[args[0]-index_map] = yc_i
                Vi[args[0]-index_map] = vc_i # new volume is total of the coelesced droplets
            if (RunTimeInputs['mode'] == "CCR"):
                theta = np.ones(len(Vi))*np.pi/4 # This is a bodge for future sorting out!!!
                                                 # as CCR is a bit unphysical in a coelescence context.
                r0 = pm.GetBase(theta, Vi/1000)
                # theta = pm.GetCAfromV(xcentres[t]/1000, r0, ZERO)  
            elif (RunTimeInputs['mode'] == "CCA"):
                theta = np.ones(len(Vi))*theta[0]
                r0 = pm.GetBase(theta, Vi/1000)
            print("V updated:", Vi)
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
            t_print=np.delete(t_print,absorbed_indices)
            print_record[t]=t_print<=t
            print("===================================")
            print("===================================")
            print("===================================")
            print("===================================")
            print("===================================")

        # gone        = Vi<=ZERO
        alive_prev=deepcopy(alive) # Note that this is only used in transients, 
                                   # you can't have coelescence and transients at the same time.

        printed = t_print<=t
        has_V   = Vi>ZERO
        alive   = np.logical_and(has_V,printed)
        
        # gone_record = np.vstack([gone_record, gone])
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
        iom.stream_hdf5_collection(f, t_i,"Time")
        iom.stream_hdf5_collection(f, V_t, volume_segment, group="Volume")
        iom.stream_hdf5_collection(f, dVdt_t, dVdt_segment, group="dVdt")
        if RunTimeInputs['box_volume']!=np.inf:
            iom.stream_hdf5_collection(f, RH_t,"Ambient_RHs")
        iom.stream_hdf5_collection(f, theta_t, theta_segment, group ="Theta")
        iom.stream_hdf5_collection(f, r0_t, radius_segment, group ="Radius")

    iom.write_hdf5_directly(t_print, 't_print', output_target)
    print("_____________________________________________")
    print("Residual Volume error = ", residual)
    print(f"Droplets took ,{t/60:.2f} mins to evaporate")
    print("_____________________________________________")
    
    return RunTimeInputs, xcentres, ycentres, print_record