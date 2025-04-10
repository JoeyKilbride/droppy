import numpy as np
import droppy as dpy

directory = input('Enter a file path: ')
pickle_name = input('Enter the name of the pickle (no ext)')
quantity = input('Enter a quantity to print: ')

data1 = dpy.load_MDTM_pickle(directory, prefix=pickle_name)
if quantity=="lifetimes":
    
    for i in range(len(data1['Volume'][0,:])):
        print(i)
        print(np.shape(data1['Volume'][0,:]))
        arg = np.argwhere(data1['Volume'][:,i]<=data1['Volume'][0,0]/1000)[0]
        print(data1['Time'][arg], end=" | ")
else:
    try:
        data1[quantity]
    except:
        print("Sorry that variable was not found.")

