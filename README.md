# 💦 DropPy

> This code uses analytic expressions to calculate the evaporation rate of an arbitrary array of droplets on a surface. It then numerically evolves the initialised array of droplets based on these calculated evaporation rates. The droplets evaporate in a diffusion-limited manner in an isothermal environment. For full details of the two analytical models implemented in "simulate.py" see: <br />
- Wray et al. 2020: doi.org/10.1017/jfm.2019.919 <br />
- Masoud et al. 2021: doi.org/10.1017/jfm.2021.785

---

## 📁 Table of Contents

- [About](#-about)
- [Features](#-features)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Screenshots](#screenshots)
- [License](#license)
- [Contact](#contact)
- [Acknowledgement](#acknowledgement)

---

## 🧠 About

The project allows you to 'simulate' the evaporation of multiple droplets on surfaces in a quasi-static environment. The spherically capped droplets can be initialised with any distribution of initial sizes and positions. The analytic theory is for diffusion limited evaporation and models the vapour interactions between the (spherical cap) droplets as they evaporate. 

---

## ✨ Features

- ✅ Droplets can evaporation in constant contact angle (CCA) mode or constant contact radius (CCR) mode. 
- ✅ The ambient humidity can be calculated dynamically based on amount of volume evaporated from the droplets, simulating an enclosed box.
- ✅ The numerical evolution of time can be Euler of a 4th Order Runge Kutta method. 
- ✅ Droplets can be set to appear at different times. For example to simulate rain landing on a window, or the order in which they are printed on the surface.

---

## ⚙️ Installation
To install the project clone (or download) the repo. 

### Using the terminal
1. Open the terminal and cd to the preferred installation directory. ie. `cd C:\User\documents` (windows), `cd /user/name/documents` (Mac).
2. Then clone with: `git clone https://github.com/JoeyKilbride/Multiple-Droplets-Theory-Models.git droppy_repo`.

### Using Zip 
1. Click green code button.
2. Download Zip.
3. Unzip to desired directory.

### Requirements

Most modern version of python 3 should work. There are quite a few required python packages which can be installed with `pip install -r requirements.txt`. The packages are as follows:
- numpy
- matplotlib
- scipy
- PIL
- importlib
- glob
- pickle
- cv2 (opencv)
- PIL
- timeit
- time

## Configuration

### Navigate into the project
`cd DTM_repo`

### (Optional) Create and activate a virtual environment
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

### Install dependencies
pip install -r requirements.txt


## How to use
1. create a folder and copy 'DTM_config_template.py' into it and rename -> 'DTM_config.py'.
   This file contains all the parameters which must be specified to run the code. It contains the specifics about where each droplet is positioned, each droplets size, liquid and gas           constants etc.
   - Change sys.path.insert(0, r'DIR OF THE CODE') -> sys.path.insert(0, r'/user/name/documents/droppy_repo'), or wherever you saved DTM_repo in installation.
   - Change prefix to a decription of the simulation ie. "example"
   A Full decription of each other variable is given as comments in the file. Should run as is, for testing. 

3. Use python to run 'initialise.py' in your prefered way.
4. Enter when prompted the directory you created in step 1 (containing the 'DTM_config.py').
5. If everything has been set correctly it will start simulating the evaporation of the specified droplets.
   By default this is a 3 droplet line.

Once the code has terminated, as long as `saving = True`, it will display some default graphs which can be closed. These graphs as well as a pickle file containing all the information from the simulation, ie. the volume of each droplet as a function of time, is saved to the folder created with the file 'DTM_config.py'. This pickle can be reloaded into python and plotted in the desired manner. 

# Contact
For questions bug reporting and help email me joey.kilbride@ed.ac.uk.
https://orcid.org/0000-0002-3699-6079

# Acknowledgement
If you make use of the code please cite: DropPy, J.J. Kilbride. (2025)








