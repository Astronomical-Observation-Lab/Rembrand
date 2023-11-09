# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 09:23:19 2023

@author: Rasjied
"""

# -*- coding: utf-8 -*-.
"""
Created on Mon Sep 12 12:51:16 2022

@author: Rasjied Sloot (m.r.sloot@uva.nl)
"""

import os
import glob
from astropy.io import fits
import numpy as np
from scipy.signal import find_peaks_cwt
from astropy.stats import sigma_clipped_stats
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from scipy import signal
from pathlib import Path
from tqdm import tqdm

main_folder = "E:/Astronomical Observation Lab/Spectroscopy Daytime Test/eShel P Cyg"

# directory = "testmap"

def make_folders(main_folder):
    folders = ['Bias', 
               'Dark', 
               'Orders', 
               'Thar', 
               'Tungsten',
               'dark_flat',
               'flat_led',
               'Flux_raw', 
               'Flux_wavelength',
               'Flux_norm',
               "Calibration", 
               "Calibration/x_pos", 
               "Calibration/wavelength_thar",
               "Calibration/x_wavelength",
               "ThAr_Table", 
               "Object",
               "Reduced"]

    for folder in folders: 

        path_name = main_folder + '/' + folder
        Path(path_name).mkdir(parents=True, exist_ok=True)

    return

make_folders(main_folder)

flat_list =  glob.glob(os.path.join(main_folder, "flat_led/*"))
darkflat_list =  glob.glob(os.path.join(main_folder, "dark_flat/*"))
tungsten_list =  glob.glob(os.path.join(main_folder, "Tungsten/*"))
bias_list=  glob.glob(os.path.join(main_folder, "Bias/*"))
thar_list = glob.glob(os.path.join(main_folder, "Thar/*"))
dark_list = glob.glob(os.path.join(main_folder, "Dark/*"))
object_list = glob.glob(os.path.join(main_folder, "Object/*.fit"))

print("Object files:",len(object_list))
print("Dark files:",len(dark_list))
print("ThAr files:",len(thar_list))

print("Flat (tungsten + LED) files:",len(flat_list))
print("Tungsten files:",len(tungsten_list))
print("DarkFlat (for tungsten) files:",len(darkflat_list))
print("Bias files:",len(bias_list))




# %%


# Create master files

bias_stack = []
for file in bias_list:
    bias_stack.append(fits.getdata(file))
masterBias = np.median(bias_stack,axis=0)

flat_stack = []
for file in flat_list:
    flat_stack.append(fits.getdata(file))
masterFlat = np.median(flat_stack,axis=0)

darkflat_stack = []
for file in darkflat_list:
    darkflat_stack.append(fits.getdata(file))
masterDarkFlat = np.median(darkflat_stack,axis=0)

tungsten_stack = []
for file in tungsten_list:
    tungsten_stack.append(fits.getdata(file))
masterTungsten = np.median(tungsten_stack,axis=0) 

thar_stack = []
for file in thar_list:
    thar_stack.append(fits.getdata(file))
masterThar = np.median(thar_stack,axis=0) 

dark_stack = []
for file in dark_list:
    dark_stack.append(fits.getdata(file))
masterDark = np.median(dark_stack,axis=0) 

object_stack = []
for file in object_list:
    object_stack.append(fits.getdata(file))
masterObject = np.median(object_stack,axis=0)

image_size=np.shape(masterTungsten)

# %%


# Calculate Read Noise

gain = fits.getheader(bias_list[0])['EGAIN']

bias1=signal.medfilt2d(np.float32(fits.getdata(bias_list[0])), kernel_size=5)
bias2=signal.medfilt2d(np.float32(fits.getdata(bias_list[1])), kernel_size=5)
read_noise = np.std(np.subtract(bias1, bias2))*gain/np.sqrt(2)
print('Read Noise (e-):',np.round(read_noise))

# %%

# Find orders with master flat frame


central_column=masterFlat[:,int(len(masterFlat[0])/2)]

mean, median, std =sigma_clipped_stats(central_column,sigma=1)
threshold =2*std +mean
central_column = gaussian_filter(central_column, sigma = 2)
peaks = find_peaks_cwt(central_column, np.arange(10, 100,2))

central_peaks=[]
for i in peaks:
    if central_column[i] > threshold:
        central_peaks.append(i)

plt.plot(central_column)
plt.axhline(threshold, label = 'threshold', color= 'red')
for i in central_peaks:
    plt.axvline(i, linestyle=":", c='gray')
plt.yscale('log')
plt.legend()
plt.show()

print('Detected orders in flat field: ',len(central_peaks))

# plot orders with numbers

central_peaks_x =np.full(len(central_peaks),int(np.shape(masterFlat)[1]/2))
order_number = np.arange(len(central_peaks))

fig,ax = plt.subplots()
ax.scatter(central_peaks_x,central_peaks, color='red')
ax.imshow(masterFlat, vmin = mean, vmax = threshold, cmap = 'gray')
for i, txt in enumerate(order_number):
    ax.annotate(txt, (central_peaks_x[i]+20,central_peaks[i]-10), color='red')
plt.show()

# %%

# Detecting orders using masterLed, this takes a while (few minutes)
peakslist_x = []
peakslist_y = []
flux = []

for i in tqdm(range(len(masterFlat[0]))):
    indexes = find_peaks_cwt(gaussian_filter(masterFlat[:,i],sigma = 3), np.arange(10, 50,5))
    # print('Detecting peaks in column',i+1,'out of',len(masterFlat[0]+1))
    for j in indexes:
        if masterFlat[:,i][j] > threshold:
            peakslist_x.append(i)
            peakslist_y.append(j)
            flux.append(masterFlat[:,i][j])

foundpeaks = np.stack((peakslist_x,peakslist_y,flux),axis=1)
print('total peaks detected:',np.shape(foundpeaks)[0])

# Plot the detected peaks
fig,ax = plt.subplots(figsize = (10,8))
# plt.figure(figsize = (10,8))
ax.scatter(foundpeaks[:,0],foundpeaks[:,1], s=foundpeaks[:,2]/1000, marker = ".", label="peaks")
ax.imshow(masterFlat, cmap='gray', label="spectrum")
for i, txt in enumerate(order_number):
    ax.annotate(txt, (central_peaks_x[i]+20,central_peaks[i]-10), color='red')
plt.title("Found peaks on the spectrum")
plt.legend()
plt.show()

# %%

# Construct peak to orders



print(np.shape(central_peaks)[0], 'orders found on central column.')
all_orders = []
for i in range(len(central_peaks)):

    order_loop = i
    # start_x = central_peaks[order_loop][0]
    start_x = image_size[1]/2
    start_y = central_peaks[i]
    
    # construct dataset within 1 order
    order_1 = [] 
    for i in range(len(foundpeaks)):
        if np.logical_and(foundpeaks[i][0] == start_x,foundpeaks[i][1] == start_y):
            order_1.append(foundpeaks[i])
        else:
            order_1.append( np.array([start_x,start_y,1000]))
    for i in range(len(foundpeaks)):
        if np.logical_and(foundpeaks[i][0] == order_1[-1][0]+1, np.abs(foundpeaks[i][1] - order_1[-1][1]) < 6 ):
            order_1.append(foundpeaks[i])
        elif  np.logical_and(foundpeaks[i][0] == order_1[-1][0]+2, np.abs(foundpeaks[i][1] - order_1[-1][1]) < 6 ):
            order_1.append(foundpeaks[i])
        elif  np.logical_and(foundpeaks[i][0] == order_1[-1][0]+3, np.abs(foundpeaks[i][1] - order_1[-1][1]) < 6 ):
            order_1.append(foundpeaks[i])
        elif  np.logical_and(foundpeaks[i][0] == order_1[-1][0]+4, np.abs(foundpeaks[i][1] - order_1[-1][1]) < 6 ):
            order_1.append(foundpeaks[i])
        elif  np.logical_and(foundpeaks[i][0] == order_1[-1][0]+5, np.abs(foundpeaks[i][1] - order_1[-1][1]) < 6 ):
            order_1.append(foundpeaks[i])
        elif  np.logical_and(foundpeaks[i][0] == order_1[-1][0]+6, np.abs(foundpeaks[i][1] - order_1[-1][1]) < 6 ):
            order_1.append(foundpeaks[i])
        elif  np.logical_and(foundpeaks[i][0] == order_1[-1][0]+7, np.abs(foundpeaks[i][1] - order_1[-1][1]) < 6 ):
            order_1.append(foundpeaks[i])
        elif  np.logical_and(foundpeaks[i][0] == order_1[-1][0]+8, np.abs(foundpeaks[i][1] - order_1[-1][1]) < 6 ):
            order_1.append(foundpeaks[i])
        elif  np.logical_and(foundpeaks[i][0] == order_1[-1][0]+9, np.abs(foundpeaks[i][1] - order_1[-1][1]) < 6 ):
            order_1.append(foundpeaks[i])
    for i in reversed(range(len(foundpeaks))):
        if np.logical_and(foundpeaks[i][0] == order_1[-1][0]-1, np.abs(foundpeaks[i][1] - order_1[-1][1]) < 6 ):
            order_1.append(foundpeaks[i])  
        elif np.logical_and(foundpeaks[i][0] == order_1[-1][0]-2, np.abs(foundpeaks[i][1] - order_1[-1][1]) < 6 ):
            order_1.append(foundpeaks[i]) 
        elif np.logical_and(foundpeaks[i][0] == order_1[-1][0]-3, np.abs(foundpeaks[i][1] - order_1[-1][1]) < 6 ):
            order_1.append(foundpeaks[i]) 
        elif np.logical_and(foundpeaks[i][0] == order_1[-1][0]-4, np.abs(foundpeaks[i][1] - order_1[-1][1]) < 6 ):
            order_1.append(foundpeaks[i]) 
        elif np.logical_and(foundpeaks[i][0] == order_1[-1][0]-5, np.abs(foundpeaks[i][1] - order_1[-1][1]) < 6 ):
            order_1.append(foundpeaks[i]) 
        elif np.logical_and(foundpeaks[i][0] == order_1[-1][0]-6, np.abs(foundpeaks[i][1] - order_1[-1][1]) < 6 ):
            order_1.append(foundpeaks[i]) 
        elif np.logical_and(foundpeaks[i][0] == order_1[-1][0]-7, np.abs(foundpeaks[i][1] - order_1[-1][1]) < 6 ):
            order_1.append(foundpeaks[i]) 
        elif np.logical_and(foundpeaks[i][0] == order_1[-1][0]-8, np.abs(foundpeaks[i][1] - order_1[-1][1]) < 6 ):
            order_1.append(foundpeaks[i]) 
        elif np.logical_and(foundpeaks[i][0] == order_1[-1][0]-9, np.abs(foundpeaks[i][1] - order_1[-1][1]) < 6 ):
            order_1.append(foundpeaks[i])             
    # remove duplicates        
    order_1=np.array(order_1)      
    order=np.unique(order_1, axis=0)
    order=np.array(order)
    order=order.astype(int)  
    print( 'Horizontal pixels detected in order',order_loop,':',np.shape(order)[0],' (', np.round(100*np.shape(order)[0]/image_size[1],1), ') %')     
    all_orders.append(order) 

plt.figure(figsize = (10,8))
for i in range(np.shape(all_orders)[0]):
    plt.scatter(all_orders[i][:,0],all_orders[i][:,1] , color='red', marker = ".")
    plt.scatter(foundpeaks[:,0],foundpeaks[:,1], s=foundpeaks[:,2]/1000, marker = ".", label="peaks")
plt.imshow(masterTungsten, cmap='gray', label="spectrum")
plt.show()

# fit polynomial curve to peaks per order

for i in range(len(all_orders)):
    fit = np.polynomial.polynomial.polyfit(all_orders[i][:,0],all_orders[i][:,1], 2)
    x_fit = []
    y_fit = []
    for x in range(len(masterTungsten[0])):
        y = 0
        # Calculate y_coordinate
          
        # Save coordinates
        x_fit.append(x)
        y_fit.append(fit[0]+fit[1]*x+fit[2]*x*x)
    order_fit = np.vstack((x_fit,y_fit))
    np.savetxt(os.path.join(main_folder, "Orders/Order{}.csv").format(i), order_fit,  delimiter=',')

print('Orders saved:', len(glob.glob(os.path.join(main_folder, "Orders/*"))))


# Now there should be a folder filled with csv files that indicate the central position of the orders.

# plot orders to check whether the detection worked:

fig,ax = plt.subplots()
ax.imshow(masterTungsten, cmap= 'gray', vmax = np.percentile(masterTungsten, 90),vmin = np.percentile(masterTungsten, 20))
for j in range(len(glob.glob(os.path.join(main_folder, "Orders/*")))):
    order = np.loadtxt(os.path.join(main_folder, "Orders/Order{}.csv").format(j), delimiter = ',')
    plt.plot(order[0], order[1], color  = 'red')
plt.ylim(image_size[0])
plt.show()

# %%



# Extract flux each frame per order and store in file

n_orders = len(glob.glob(os.path.join(main_folder, "Orders/*")))


for j in range(n_orders):
    print ("extracting order {}".format(j))
    order = np.loadtxt(os.path.join(main_folder, "Orders/Order{}.csv").format(j), delimiter = ',')
    order = order.astype(int)
    
    spectrum_thar=[]
    spectrum_tungsten=[]
    spectrum_darkflat=[]
    spectrum_bias=[]
    spectrum_dark=[]
    spectrum_object=[]
    SNR = []
    
    width =20
    for i in range(len(order[0])):
        
        intensity_tungsten = np.sum(masterTungsten[:,i][order[1][i]-width:order[1][i]+1+width])
        spectrum_tungsten.append(intensity_tungsten)
        intensity_darkflat = np.sum(masterDarkFlat[:,i][order[1][i]-width:order[1][i]+1+width])
        spectrum_darkflat.append(intensity_darkflat)
        intensity_bias = np.sum(masterBias[:,i][order[1][i]-width:order[1][i]+1+width])
        spectrum_bias.append(intensity_bias)
        intensity_thar = np.sum(masterThar[:,i][order[1][i]-width:order[1][i]+1+width])
        spectrum_thar.append(intensity_thar)
        intensity_dark = np.sum(masterDark[:,i][order[1][i]-width:order[1][i]+1+width])
        spectrum_dark.append(intensity_dark)
        intensity_object = np.sum(masterObject[:,i][order[1][i]-width:order[1][i]+1+width])
        spectrum_object.append(intensity_object)
        
        S_dark = np.abs(intensity_dark-intensity_bias)
        flux = intensity_object-intensity_dark
        noise = np.sqrt(flux+(2*width+1)*(S_dark + read_noise**2))
        
        SNR.append(flux/noise)
        
        
    # SNR = 
    data_merged = np.vstack((spectrum_thar,spectrum_tungsten,spectrum_bias,spectrum_dark,spectrum_object,SNR, spectrum_darkflat))
    
    np.savetxt(os.path.join(main_folder, "Flux_raw/data_raw_order_{}.csv").format(j), data_merged,  delimiter=',')
    
    

    plt.figure(figsize = (10,8))
    plt.title("Order {} (scaled)".format(j))
    plt.plot(spectrum_object, label='Object')
    plt.plot(spectrum_thar,label='ThAr')
    plt.plot(spectrum_tungsten,label='Tungsten')
    plt.plot(spectrum_bias, label='Bias')
    plt.plot(spectrum_dark, label='Dark')
    # plt.plot(SNR, label='SNR')
    # plt.ylim(0,1.1)
    plt.legend()
    plt.show()

# formatting of exported datasets: Thar,Tungsten,Bias,Dark,Object, SNR, darkflat

# %%

# The following code can be used to load the flux data from an order:
# For example if you would like the flux of the object in order 3:

N_order = 24
data_order_N = np.loadtxt(os.path.join(main_folder, "Flux_raw/data_raw_order_{}.csv").format(N_order),  delimiter=',')
plt.plot(data_order_N[4])



