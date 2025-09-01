#!/usr/bin/env python #
"""\
This file contains physicy functions used to run the code. 

"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import scipy
import warnings
from copy import deepcopy


def normalise(arr, by='max'):
    """by: max, min or mean"""
    if by=='mean':
        arr_norm = arr/np.mean(arr)
    elif by=='min':
        arr_norm = arr/np.min(arr)
    else:
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

def circular_sort(X, Y):
    """Function sorting x,y points into circles from centre to edge."""
    rdx = np.sqrt((X-np.mean(X))**2+(Y-np.mean(Y))**2)
    sort = np.argsort(rdx)
    print(sort)
    Xs = np.array(X)[sort]
    print("Xs= ",len(Xs))
    Ys = np.array(Y)[sort]
    plt.scatter(Xs,Ys,c='k')
    plt.show()
    print("Xs: ",len(Xs))
    Xss = []
    Yss = []
    rdx = np.sqrt((Xs-np.mean(Xs))**2+(Ys-np.mean(Ys))**2)
    rdx_min = np.min(np.sqrt((Xs[0]-Xs[1:])**2+(Ys[0]-Ys[1:])**2))
    inc = rdx_min/2
    for b in np.arange(0,np.max(rdx)*1.01,inc):
        print(b)
        band = np.where(np.logical_and(rdx>=b,rdx<b+inc))
        if len(band[0])>0:
            Xm = Xs[band]
            Ym = Ys[band]
            theta  = np.arctan2(Ym,Xm)
            theta_sort = np.argsort(theta)
            plt.scatter(Xs, Ys, c='k')
            plt.scatter(Xm,Ym, c=theta, cmap='hot')
            theta = np.linspace(0, 2 * np.pi, 300)
            ellipse_x = b * np.cos(theta)
            ellipse_y = b * np.sin(theta)
            plt.plot(ellipse_x, ellipse_y, color='red', linewidth=2, label="Ellipse Boundary")
            ellipse_x = (b+inc) * np.cos(theta)
            ellipse_y = (b+inc) * np.sin(theta)
            plt.plot(ellipse_x, ellipse_y, color='red', linewidth=2, label="Ellipse Boundary")
            plt.show()
            Xss = Xss + list(Xm[theta_sort])
            Yss = Yss + list(Ym[theta_sort])
    
    print("Xss: ", len(Xss))
    plt.scatter(Xss,Yss, c=range(0,len(Yss)), cmap='hot')
    for idx, i in enumerate(Xss):
        plt.text(Xss[idx],Yss[idx], str(idx))
    plt.show()
    return np.array(Xss), np.array(Yss)

def rectangle_points(length, width, s):
    """
    Generate a grid of points inside a rectangle.

    Parameters:
        length (float): Length of the rectangle (x-direction).
        width (float): Width of the rectangle (y-direction).
        s (float): Spacing between points.

    Returns:
        (np.ndarray, np.ndarray): Arrays of x and y coordinates.
    """
    x_coords = np.arange(0, length + s, s)
    y_coords = np.arange(0, width + s, s)

    X, Y = np.meshgrid(x_coords, y_coords, indexing="xy")

    return X.ravel(), Y.ravel()


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
            
def depositTriangle(N,s,a):
    xs = s*a
    ys = s*a*np.sqrt(3)/2
    total = int(N*(N+1)/2)
    X=np.empty(total)
    Y=np.empty(total)
    counter =0
    Ny=N
    for j in range(N):
        for i in range(Ny):
            print(counter)
            print("i= ",i, " j=",j)
            X[counter]=(i*xs)+j*(xs/2)
            Y[counter]=j*ys
            counter =counter+1
        Ny=Ny-1
    return X, Y

def hexagonal_grid_in_ellipse(a, b, s):
    """
    Generate hexagonal grid points inside an ellipse centered at (0,0) with semi-axes a and b.
    
    Parameters:
        a (float): Semi-major axis of the ellipse.
        b (float): Semi-minor axis of the ellipse.
        s (float): Spacing between points.
        
    Returns:
        tuple of lists: Two lists containing x and y coordinates of points inside the ellipse.
    """
    x_points = []
    y_points = []
    x_out = []
    y_out = []
    dx = s
    dy = np.sqrt(3) / 2 * s  # Vertical spacing for hexagonal grid
    
    # Define grid boundaries
    x_range = np.arange(-a-3*(dx/2), a + 3*(dx/2), dx)
    y_range = np.arange(-b - 3*(dy/2), b + 3*(dy/2), dy)
    xm = np.mean(x_range)
    ym = np.mean(y_range)

    # insures the points are symmetrix about
    x_range = x_range - xm  # y=0
    y_range = y_range - ym # x=0

    for i, y in enumerate(y_range):
        offset = (i % 2) * (s / 2)  # Offset every other row
        for x in x_range:
            x_pos = x + offset
            x_points.append(x_pos)
            y_points.append(y)
    
    xm = np.mean(x_points)
    ym = np.mean(y_points)
    r_closest = np.argmin(np.sqrt((x_points-xm)**2+(y_points-ym)**2))
    x_points = x_points - x_points[r_closest]
    y_points = y_points - y_points[r_closest]
    for pdx, point in enumerate(x_points):
        if (point**2 / a**2 + y_points[pdx]**2 / b**2) <= 1.00:  # Inside ellipse condition
            x_out.append(point)
            y_out.append(y_points[pdx])

    return np.array(x_out), np.array(y_out)

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

def Masoud_fast(x, y, a, dVdt_iso, CA):
    """Calculating Masoud et al. 2020 theoretical evaporation
    rates for multiple droplets.
    Returns droplet evaporation rates in L/s. """
    
    def intA(x, CA):
        return (1+ (np.cosh((2*np.pi-CA)*x)/np.cosh(CA*x) ) )**-1
    
    def intA2(CA):
        return scipy.integrate.quad(intA, 0, 100, args=(CA))[0]

    # def intA2_lowCA(CA):
    #     return scipy.integrate.quad(intA, 0, 100, args=(CA))[0]

    def intB(x, CA):
        return ((1+ (np.cosh((2*np.pi-CA)*x)/np.cosh(CA*x) ) )**-1) *x**2
    
    def intB2(CA):
        return scipy.integrate.quad(intB, 0, 100, args=(CA))[0]
    
    # def intB2_lowCA(CA):
    #     return scipy.integrate.quad(intB, 0, 100, args=(CA))[0]
    
    
    VintA= np.vectorize(intA2)
    A=VintA(CA)
    VintB= np.vectorize(intB2)
    B=VintB(CA)
         
    Rij = a/np.sin(CA)
    hij = Rij-(a/np.tan(CA))
    zi = ((3/4) * ((2*Rij-hij)**2/(3*Rij-hij)) ) - (Rij-hij) #Rij - hij/3
    z = np.abs(zi[:, None] - zi[None, :]) # difference between geometric centres in z

    x_diff = x[:,None]-x[None,:]
    y_diff = y[:,None]-y[None,:]
    r = np.sqrt(x_diff**2+y_diff**2)
    np.fill_diagonal(r,1)

    a_b = a[:,None]
    A_b = A[:,None]
    B_b = B[:,None]

    X = 4*(a_b/r)*A_b + (A_b-4*B_b)*((a_b**3*(r**2-3*z**2))/(r**5))      
    np.fill_diagonal(X,1)
    
    lu, piv = scipy.linalg.lu_factor(X)
    dVdt=scipy.linalg.lu_solve((lu,piv),dVdt_iso)*1000

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
               
                X[idx,jdx] = (2/np.pi) * np.arcsin(a[idx]/r[idx,jdx])   # equation (3.2) from the paper - poly-volume droplets
                    #X[idx,jdx] =  (2/np.pi) * np.arcsin(a[idx]/r[idx,jdx])
    #Y = linalg.inv(X)        # inverse of the matrix...next step, calculate the flux
    #dVdt = dot(Y,dVdt_iso)*1000 # L/s: dot prod sums up all contributions - lab book 03/09/21
    #dVdt=scipy.linalg.solve(X, dVdt_iso)*1000  # 1.62X  faster than above # 3.6X faster
    dVdt=scipy.linalg.lu_solve(scipy.linalg.lu_factor(X),dVdt_iso)*1000

    return dVdt # return theoretical flux values


def getIsolated(csat, H, Rb, CA, rho_liquid, D, Mm, sigma, T):
    """Returns the evaporation rate for an isolated droplet at a 
    temperature (oC), humidity (H) and for a base radius (Rb) and contact angle (CA) and
    liquid density (rho_liquid)."""

    if np.any(CA==0):
        r = np.where(CA==0,Rb,Rb/np.sin(CA))
        phi_sat = kohler(0,Mm,sigma,T,rho_liquid, r)
    else:
        phi_sat = kohler(0,Mm,sigma,T,rho_liquid, Rb/np.sin(CA))

    dmdt_env = D*csat*(phi_sat-H) # Calculate envionmental component of flux.
    f_theta = 2/np.sqrt(1+np.cos(CA)) # Hu 2014 (0 to pi) was: 0.27*(CA**2)+1.30 0 to pi/2
    dmdt_geom = np.pi*Rb*f_theta 
    dmdt=dmdt_env*dmdt_geom 
    dVdt_isolated=-(dmdt/rho_liquid) # m3/s - Convention is to have -dVdt as evaporation
    
    return dVdt_isolated

def Psat(A,B,C,T):
    """Saturation vapour pressure using the Antoine equation.
        A,B,C: Antoine constants for fluid.
        T: Temperature (oC)"""
    psat = 10**(A-(B/(C+T)))*(101325/760)
    return psat

def gas_density(ABCs, mms, phis, T):
    """Get density of mixtures, T in degrees C."""
    ps = np.empty([len(mms)])
    for idx, i in enumerate(ABCs):
        ps[idx] = Psat(*i,T)
    rho = (((101325+np.sum(-1*(ps*phis)))*0.02897)+np.sum(phis*ps*mms))/(8.314*(T+273.15))
    return rho

def kohler(n,Mm,sigma,T,rho,radius):
    """Kohler theory calculating the vapour pressures variation due to:
    The Kelvin effect and The Raoult effect."""
    Di = radius*2 # diameter (m)
    kelvin = (4*Mm*sigma)/(8.314*(T+273.15)*rho*Di)
    Raoult = (6*n*Mm)/(np.pi*rho*Di**3)
    p_eq = np.exp(kelvin-Raoult)
    return p_eq

def ideal_gas_law(P,T,Mm):
    """Calculates the vapour density (kg.m-3) using the ideal gas law.
    T (oC) Temperature
    Mm (kg.mol-1) molar mass
    P (Pa) vapour pressure"""
    nbyV=P/(8.314*(T+273.15))
    concentration = nbyV*Mm
    return concentration


def dynamic_humidity(box_volume,molar_mass,A,B,C,T,rho, V_evap):
    psat = Psat(A,B,C, T-273.15)
    R = 8.314
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
