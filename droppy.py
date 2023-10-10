from __future__ import division
import numpy as np
import matplotlib
from matplotlib.collections import PatchCollection
from matplotlib.patches import Wedge
from matplotlib.axes._axes import _log as matplotlib_axes_logger
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.integrate import quad
from PIL import ImageColor
import scipy
import os
import sys
import importlib
import glob
import warnings
import pickle
matplotlib_axes_logger.setLevel('ERROR')
from copy import deepcopy
#from scipy.special import legendre
#from matplotlib.animation import FFMpegWriter
#import itertools
#import pickle
#import math
#import time
#import networkx as nx
#from scipy.special import legendre


# General functions ******************************************************
def normalise(arr):
    arr_norm = arr/np.max(arr)
    return arr_norm

def get_radial_position(X, Y, xref, yref):
    """Returns coordinate X,Y radial distance from centre of the array."""
    radial = np.sqrt(((X-xref)**2+(Y-yref)**2))
    return radial 

def TouchingCircles(cx,cy,r, ca):
    """filters touching circles from polydisperse array.
    Returns boolean mask of touching circles to filter original arrays."""
    print("Raw droplets: ", len(cx))
    touching=np.ones([len(cx)],dtype=bool) 
    for idx, i in enumerate(cx): 
        print(idx)
        for jdx, j in enumerate(cx):
        
            if touching[idx]==0:
                continue
            else:
                s=np.sqrt((i-j)**2+(cy[idx]-cy[jdx])**2)
                if s==0:
                    #print("same droplet")
                    #print("idx: "+str(idx)+" jdx: ", str(jdx))
                    continue
                elif s<(r[idx]+r[jdx]):
                    print("idx: "+str(idx)+" jdx: ", str(jdx))
                    if r[idx]>r[jdx]:
                        touching[jdx]=0
                        #print("Removing jdx: "+ str(jdx))
                        #print("removed: ",count_nonzero(touching))
                    else:
                        touching[idx]=0
                        #print("Removing idx: "+str(idx))
                        #print("removed: ",count_nonzero(touching))
                        continue
                else:
                    touching[idx]=1
    print("filtered droplets: ", len(cx[touching]))
    CX=cx[touching]
    CY=cy[touching]
    Rb=r[touching]
    CA=ca[touching]  
    return CX, CY, Rb, CA

# IO functions ************************************************************
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
    #import MDL_config as config
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
        target = glob.glob(os.path.join(directory,prefix))[0]
        if os.path.getsize(target)>0:
            with open(target, 'rb') as handle:
                input_dict = pickle.load(handle)
    except:
        #print("Trying .pkl")
        if prefix==None:
            prefix = "MDL_*.pkl"
        else:
            prefix = prefix[:4]+ ".pkl"
        target = glob.glob(os.path.join(directory,prefix))[0]        
        if os.path.getsize(target)>0:
            with open(target, 'rb') as handle:
                input_dict = pickle.load(handle)

    return input_dict

def load_MDTM_pickle(directory, prefix=None):
    """Load a droplet theory prediction pickle.
        prefix (Optional): specify a specific filename (no ext)."""
    #print("Trying with .pickle")
    if prefix==None:
        prefix1 = "MDTM_*.pickle"
    else:
        prefix1 = prefix + ".pickle"
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

# Visualisation functions **************************************************

def add_bias(xb,yb,x,y, bias_gradient):
    b_ang = np.arctan2(yb,xb)
    theta_2 = np.arctan2(y,x)
    theta_1 = (b_ang-theta_2)

    rd = get_radial_position(x, y, 0, 0) # distance to droplet
    bd = rd*np.cos(theta_1) # distance along bias direction
    radial_bias = np.sqrt(xb**2+yb**2)
    mb = (-bias_gradient/radial_bias)
    bias = (bd*mb)+1
    return b_ang, mb, bias[0]

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
    
def ReportResults(Results, cmap_name):
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
    
    ND = Results["RunTimeInputs"]['DNum']
    xc = np.mean(Results["RunTimeInputs"]["xcentres"])
    yc = np.mean(Results["RunTimeInputs"]["ycentres"])

    t_evap = np.empty([ND])
    for pdx in range(ND):
        name = "D"+str(pdx)
        ax_V.plot(Results['Time'], Results['Volume'][:,pdx]*1e6, label=name, marker="o")
        elem=list(np.nonzero(Results['Volume'][:,pdx]))[0][-1] # finds element where V is first zero.
        t_evap[pdx]=Results['Time'][elem]
        #print("evap time ", t_evap[pdx])
        dist_2_centre = np.sqrt((Results["RunTimeInputs"]["xcentres"][pdx]-xc)**2+(yc-Results["RunTimeInputs"]["ycentres"][pdx])**2)
        #ax_tevap.scatter(dist_2_centre,t_evap[pdx])
        #print("Inital Contact Angle: ", Results["RunTimeInputs"]["CA"][pdx])
        #print("Inital CX: ", Results["RunTimeInputs"]["xcentres"][pdx])
        #print(Results['Volume'][:,pdx]*1e6)
        if Results['RunTimeInputs']['mode']=="CCR":
            ax_idx.plot(Results['Time'], Results['Theta'][:,pdx], label=name, marker="o")
        else:
            ax_idx.plot(Results['Time'], Results['Radius'][:,pdx], label=name, marker="o")
            
        ax_dVdt.plot(Results['Time'], Results['dVdt'][:,pdx]*1e6, label=name)
    #print("t_evap = ", t_evap)
    #print("Mean calculation time: ",mean(Results['Calc_Time']))
    #print("Total calculation time: ",sum(Results['Calc_Time']))
    s_dt = ax_dt.scatter(Results['RunTimeInputs']['xcentres'],Results['RunTimeInputs']['ycentres'],\
         c=normalise(t_evap), cmap=cmap_name, vmin=0, vmax=1)
    ax_dt.set_aspect('equal', adjustable='box')
    fig_dt.colorbar(s_dt, ax=ax_dt,  orientation='horizontal')
    #print("Number of droplets: ",Results['RunTimeInputs']['DNum'])
    ax_V.set_ylabel("V (\u03BC"+ "L)")
    ax_V.set_xlabel("Time (s)")
    if Results['RunTimeInputs']['DNum']<15:
        ax_V.legend()
        ax_idx.legend()
        ax_dVdt.legend()
        
    ax_idx.set_xlabel("Time (s)")
    if Results['RunTimeInputs']['mode']=="CCR":
        ax_idx.set_ylabel("\u03B8 (\u00b0)")
    else:
        ax_idx.set_ylabel(r"$R_{b}$ (mm)")
    ax_dVdt.set_xlabel("Time (s)")
    ax_dVdt.set_ylabel("dV/dt (\u03BC"+ "L/s)")
    fig_V.savefig(os.path.join(Results['RunTimeInputs']['Directory'],Results['RunTimeInputs']['Filename']+"_V.png"))
    fig_idx.savefig(os.path.join(Results['RunTimeInputs']['Directory'],Results['RunTimeInputs']['Filename']+"_CA.png"))
    fig_dVdt.savefig(os.path.join(Results['RunTimeInputs']['Directory'],Results['RunTimeInputs']['Filename']+"_dVdt.png"))
    fig_dt.savefig(os.path.join(Results['RunTimeInputs']['Directory'],Results['RunTimeInputs']['Filename']+"_drytime_heatmap.png"))
    #fig_tevap.savefig(os.path.join(Results['RunTimeInputs']['Directory'],Results['RunTimeInputs']['Filename']+"_tevap.png"))
    Resultsfile = open(os.path.join(Results['RunTimeInputs']['Directory'],"MDTM_"+Results['RunTimeInputs']['Filename']+'.pkl'), 'wb')
    Results['t_evap'] = t_evap
    pickle.dump(Results, Resultsfile)
    Resultsfile.close()
    plt.show()
    return


def depositSquare(N,s,ca,Rb):
    # N, number of droplets NxN
    # s, separation (*Rb) 2=touching
    # Rb, base radius of droplets in array
        
    Edge= (N-1)*(Rb*s)/2
    x=np.linspace(-Edge,Edge,N)
    y=np.linspace(-Edge,Edge,N)
    X,Y=np.meshgrid(x,y)
    xcentres=X.flatten()
    ycentres=Y.flatten()
    contact_angle=np.ones(len(xcentres))*ca
    base_radius=np.ones(len(xcentres))*Rb
  
    return xcentres, ycentres, contact_angle, base_radius

def depositLine(N,s,ca,Rb):
    Edge= Rb*(N*s-s)/2
    xcentres=np.linspace(-Edge,Edge,N)
    ycentres=np.zeros(len(xcentres))
    contact_angle=np.ones(len(xcentres))*ca
    base_radius=np.ones(len(xcentres))*Rb
    return xcentres, ycentres, contact_angle, base_radius

def depositHexagon(Nr, s, Rb):
    """Deposit a filled hexagon.
    N number of rings around centre droplet.
    s in units of Rb."""
    N_edge=Nr
    N_middle=(2*Nr)-1
    ldots = np.hstack([np.linspace(N_edge,N_middle,N_edge,dtype=int),np.linspace(N_middle-1,N_edge,N_edge-1,dtype=int)])
    x = np.empty([0])
    y = np.empty([0])
    direction=-1
    direction2=1
    ys = np.sqrt(s**2-(s/2)**2)*Rb
    xval=0
    for layer in range(N_middle):
        direction=direction*-1 # change direction
        for dot in range(ldots[layer]-1):
            x = np.hstack([x,xval])

            y = np.hstack([y,layer*ys])
            xval = xval + s*Rb*direction
        x = np.hstack([x,xval])
        y = np.hstack([y,layer*ys])    
        if layer<N_edge-1:
            direction2=deepcopy(direction)
        else:
            direction2=-deepcopy(direction)
        xval = xval+direction2*(s/2)*Rb

    return x, y, Rb*np.ones(len(x))
            

def depositTriangle_NOTWORKING(N,s,ca,Rb):
    side=(N-1)*s
    xshift = s
    yshift = s
    
    Nt=int((N*(N+1))/2) # total number of droplets in triangle
    xcentres=np.zeros(Nt)
    ycentres=np.zeros(Nt)
    Nx=N
    x0=0
    counter = 0
    for ys in range(N):
        print(ys)
        for xs in range(Nx):
            xcentres[counter] = x0+xshift*xs
            ycentres[counter] = yshift*ys
            counter = counter + 1
        Nx=Nx-1
        x0=x0+s/2
    print(xcentres)
    print(side/2)
    xcentres = (xcentres-(side/2))*Rb
    ycentres = (ycentres-(side/2))*Rb
    contact_angle=np.ones(len(xcentres))*ca
    base_radius=np.ones(len(xcentres))*Rb
    return xcentres, ycentres, contact_angle, base_radius

# Spherical cap geometry functions ***********************************************

def GetVolumeCA(CA, r_base):
    """Calculates volume of a spherical cap from Contact angle (rads)
    and base radius (m). Returns V in L."""
    V=(np.pi/3)*((r_base/np.sin(CA))**3) *(2+np.cos(CA))*((1-np.cos(CA))**2)
    V=V*1000 # m3 -> L
    return V
def GetBase(CA, V):
    """Calculates Rb (m) of a spherical cap from Contact angle (rads)
    and volume (m^3). Returns Rb in m."""
    Rb=((V*np.sin(CA)**3)/((np.pi/3)*(2+np.cos(CA))*((1-np.cos(CA))**2)))**(1/3)
    return Rb

def GetVolume(height, r_base):
    #print("height=",height)
    #print("rbase=",r_base)
    V=(1/6)*np.pi*height*(3*r_base**2+height**2) # m^3
    #print("Volume in function = ",V)
    V=V*1000 # Convert to litres
    return V

def BondNumber(del_rho,L,gamma):
    """Returns the Bond Number of a droplet."""
    Bo=(del_rho*9.81*L**2)/(gamma)
    return Bo

def GetCA(h,Rb):
    """Returns the contact angle of a the spherically capped droplet
        of height h and base radius Rb."""
    return 2*np.arctan(h/Rb)

def GetCAfromV(V, Rb, zero):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Rb=np.array(Rb)
        V=np.array(V)
        height= GetHeight(Rb, V)
        r = (Rb**2+height**2)/(2*height)    
        prefactor=(np.pi/3)*r**3
        A =  np.ones(len(V))
        B =  0*np.ones(len(V))
        C = -3*np.ones(len(V))
        D =  2-(V/prefactor)
        p=np.vstack([A,B,C,D]*prefactor)
        cnum=len(p[0,:])
        CA=np.empty([cnum])
        for i in range(cnum):
            if any(p[:,i]==np.Inf*np.ones(4)): 
                CA[i]=0
            else:
                rts=np.roots(p[:,i])
                CA[i] = abs(np.arccos(rts[-1]))
    return CA

def GetHeight(Rb, V):
    """Calculates the height (output) of a spherical cap droplet
        from the volume (input 2) and base radius (input 1).""" 
    Rb=np.array(Rb)
    V=np.array(V)

    A=np.ones(len(Rb))
    B=np.zeros(len(Rb))
    C=3*Rb**2
    D=6*V/np.pi # V is returned from GetVolume in Litres, convert to m^3
    p=np.vstack([A,B,C,D])
    hnum=len(p[0,:])
    heights=np.empty([hnum])
    for i in range(hnum):
      #  print("p coeffs",p[:,i])
        rts=np.roots(p[:,i])
     #   print("roots",roots)
        heights[i] = abs(rts[-1])
    #print("height=",heights)
    return heights

def GetHeightCA(r, CA):
    """Calculates the height (output) of a spherical cap droplet
    from the CA (input 2) and base radius (input 1).""" 
    h=r*np.tan(CA/2)
    return h

def GetRoC(r_base,h):
    return (h**2+r_base**2)/(2*h)

# Droplet evaporation functions *********************************

def Masoud(x, y, a, dVdt_iso, CA):
    """Calculating Masoud et al. 2020 theoretical evaporation
    rates for multiple droplets.
    Returns droplet evaporation rates in L/s. """

    def intA(x, CA):
        return (1+ (np.cosh((2*np.pi-CA)*x)/np.cosh(CA*x) ) )**-1
    
    def intA2(CA):
        return quad(intA, 0, 100, args=(CA))[0]

    def intB(x, CA):
        return (1+ (np.cosh((2*np.pi-CA)*x)/np.cosh(CA*x) ) )**-1 *x**2
    
    def intB2(CA):
        return quad(intB, 0, 100, args=(CA))[0]

    VintA= np.vectorize(intA2)
    A=VintA(CA)
    VintB= np.vectorize(intB2)
    B=VintB(CA)
  
    N = np.size(x)
    r=np.empty([N,N])           # set-up empty array for inter-droplet distance
    X=np.empty([N,N])


    for idx, i in enumerate(x):
            for jdx, j in enumerate(x):
                r[idx,jdx]=np.sqrt((i-j)**2+(y[idx]-y[jdx])**2)
                #print("r=",r)
                if idx == jdx :     # the diagonal terms should be one
                    X[idx,jdx] = 1
                else:
                    #print("test including the second term here")
                    X[idx,jdx] =  4*(a[idx]/r[idx,jdx])*A[idx] #+ (A-4*B)*((a[idx]**3*(r[idx,jdx]**2-3*z[idx]**2))/r[idx,jdx]**5) # equation (3.1) from Masoud et al. 2021

                    
    dVdt=scipy.linalg.lu_solve(scipy.linalg.lu_factor(X),dVdt_iso)*1000
    #dVdt=scipy.linalg.solve(X, dVdt_iso)*1000 
    
    #Y = linalg.inv(X)        # inverse of the matrix...next step, calculate the flux
    #dVdt = dot(Y,dVdt_iso)*1000 # L/s: dot prod sums up all contributions - lab book 03/09/21
    
    return dVdt

def WrayFabricant(x,y,a,dVdt_iso):
    """Calculating Wray et al. 2020 theoretical
    evaporation rates for multiple droplets.
    Returns droplet evaporation rates in L/s."""
    # x =array(x,dtype=float) # force numpy array
    # y =array(y,dtype=float)
    # a =array(a,dtype=float)

    N = np.size(x)
    r=np.empty([N,N])           # set-up empty array for inter-droplet distance
    X=np.empty([N,N])           # set-up empty array for Flux calculation
    #print("x=",x)
    for idx, i in enumerate(x):
       
        for jdx, j in enumerate(x):
           
            #print((a[jdx]/a[idx])**2)
            r[idx,jdx]=np.sqrt((i-j)**2+(y[idx]-y[jdx])**2)
            if idx == jdx :     # the diagonal terms should be one
                X[idx,jdx] = 1
            else:
               
                X[idx,jdx] = (2/np.pi) * (a[jdx]/a[idx])**2 * np.arcsin(a[idx]/r[idx,jdx])   # equation (3.2) from the paper - poly-volume droplets
                    #X[idx,jdx] =  (2/np.pi) * np.arcsin(a[idx]/r[idx,jdx])
    #Y = linalg.inv(X)        # inverse of the matrix...next step, calculate the flux
    #dVdt = dot(Y,dVdt_iso)*1000 # L/s: dot prod sums up all contributions - lab book 03/09/21
    #dVdt=scipy.linalg.solve(X, dVdt_iso)*1000  # 1.62X  faster than above # 3.6X faster
    dVdt=scipy.linalg.lu_solve(scipy.linalg.lu_factor(X),dVdt_iso)*1000

    return dVdt # return theoretical flux values


def getIsolated(Temperature, H, Rb, CA, rho_liquid):
    """Returns the evaporation rate for an isolated droplet at a 
    temperature (oC), humidity (H) and for a base radius (Rb) and contact angle (CA) and
    liquid density (rho_liquid)."""
    ###DCv = (3e-48)*(Temperature)**(19.5) # Constants from Temperature.
    #D=0.225e-4*(Temperature/273.15)**1.8 # Old D calculation not as accurate changed 14/01/2022
    D=diffusion_coeff(Temperature)
    #print("D = ",D)
    T_kelvin = Temperature +273.15
    Cv=saturation_vapour_density(T_kelvin)
    #print("\nCv = ",Cv)
    #DCv = D*Cv#(2.037e-12/(Temperature**8.2))*exp(77.345+0.0057*Temperature-(7235/Temperature)) 
    dmdt_env = D*Cv*(1-H) # Calculate enviomental component of flux.
    dmdt_geom = np.pi*Rb*(0.27*(CA**2)+1.30)
    dmdt=dmdt_env*dmdt_geom 
    dVdt_isolated=-(dmdt/rho_liquid) # m3/s - Convention is to have -dVdt as evaporation
    
    return dVdt_isolated

def Psat(A,B,C,T):
    """Saturation vapour pressure using the Antoine equation.
        A,B,C: Antoine constants for fluid.
        T: Temperature (oC)"""
    psat = 10**(A-(B/(C+T)))
    return psat

def dynamic_humidity(box_volume,molar_mass,A,B,C,T,rho, V_evap):
    psat = Psat(A,B,C,T-273.15)*(101325/760)
    R = 8.314
    molar_mass = 0.01801528 # kg/mol
    n = (psat*box_volume)/(R*T)
    msat = n*molar_mass
    RH_rise = (rho*V_evap/1000)/msat
    return RH_rise

def saturation_vapour_density(T_k):
    """Saturation vapour density kg/m3 at temperature T (kelvin)."""
    return 0.0022*np.exp(77.345+0.0057*T_k-(7235/T_k))/T_k**9.2 # i think kgm-3



def diffusion_coeff(ToC):
    """from S3.3 doi.org/10.1016/B978-0-12-386910-4.00003-2."""
    return (1+0.007*ToC)*21.2e-6 
# def Tonini(CX,CY,Rb,CA):

#     psi0 = np.pi-CA
#     h = GetHeightCA(Rb, CA)
    
#     def Legendre(x,n):
#         P_n = legendre(n)
#         P_n_x = P_n(x)
#         return P_n_x
#     #https://mathworld.wolfram.com/ConicalFunction.html 
#     def integrand(tau, xi, psi0, psi): 
#         Mehler_func = Legendre(np.cosh(xi) ,complex(-1/2,tau))
#         vals = Mehler_func*((np.cosh((np.pi-psi0)*tau)*np.cosh(psi*tau))/(np.cosh(np.pi*tau)*np.cosh(psi0*tau)))
#         return vals
    
#     def calculate_integral(xi, psi0, psi):
#         return quad(integrand, 0, np.inf, args=(xi, psi0, psi))[0]
    
    

#     alpha=0# i think?
#     gamma=0# could be??
#     N = np.size(CX)
#     X=np.empty([N,N])
#     print(CX)
#     for idx, i in enumerate(CX):
#         for jdx, j in enumerate(CX):
            
#             psi0 = np.pi-CA[idx]
#             if idx == jdx :     # the diagonal terms should be one
#                 X[idx,jdx] = 1
#             else:
               
#                 x = CX[idx]-CX[jdx]
#                 y = CY[idx]-CY[jdx]
#                 z = h[idx]
#     # from http://www.fractalforums.com/theory/toroidal-coordinates/                
#                 xi = np.sqrt((np.sqrt(x**2+y**2)-alpha)**2+z**2)
#                 psi = np.arctan((z)/(np.sqrt(x**2+y**2)-alpha))-gamma
#                 #phi = np.arctan(y/x)
#                 Theta = np.cosh(xi)-np.cos(psi)
#                 Sigma = np.sqrt(2)*Theta**(1/2)
#                 calc_integrals= np.vectorize(calculate_integral)
#                 X[idx,jdx]=Sigma*calc_integrals(xi, psi0, psi)
#     return X
