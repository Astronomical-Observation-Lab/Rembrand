# -*- coding: utf-8 -*-
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

from tqdm import tqdm

main_folder = "E:/Astronomical Observation Lab/Spectroscopy Daytime Test/eShel P Cyg"


# %%

# _____________________________________________________
# ______________Wavelength calibration_________________
# _____________________________________________________

cal_lines = np.loadtxt(os.path.join(main_folder, "ThAr_Table/ThAr_tabel3.csv"),  delimiter=',')

x_offset = 0
cal_lines[:,1] = cal_lines[:,1] + x_offset

# input position of the ThAr line with wavelength 6604.8534 angstrom

reference_line = np.argwhere(cal_lines[:,3]==6604.8534)[0][0]
reference_line_x = cal_lines[reference_line][1]
reference_line_y = cal_lines[reference_line][2]
reference_line_order = cal_lines[reference_line][0]

# order where the reference line is detected:

print('refernce line:','x: ',reference_line_x,'y: ',reference_line_y,'order: ',reference_line_order)

thar_list = glob.glob(os.path.join(main_folder, "Thar/*"))
thar_stack = []
for file in thar_list:
    thar_stack.append(fits.getdata(file))
masterThar = np.median(thar_stack,axis=0) 

fig,ax = plt.subplots(figsize=(16,9))
for j in range(len(glob.glob(os.path.join(main_folder, "Orders/*")))):
    order = np.loadtxt(os.path.join(main_folder, "Orders/Order{}.csv").format(j), delimiter = ',')
    ax.plot(order[0], order[1], color  = 'red')
ax.scatter( cal_lines[:,1],cal_lines[:,2] , s=150, facecolors='none', edgecolors='g')
ax.scatter( reference_line_x,reference_line_y , s=200, facecolors='none', edgecolors='b')
ax.imshow(masterThar, vmin=np.median(masterThar), vmax=np.median(masterThar)+500, cmap='gray')
plt.show()

# in which order is the reference line? (blue line in previous plot. The top order is order 0.)

reference_order = 2

# calibration per order and add to existing dataset:

# formatting of imported datasets: Thar,Tungsten,Bias,Dark,Object, SNR, darkflat

def gaussian(x, amp, cen, wid):
    return amp * np.exp(-(x-cen)**2 / wid)

n_orders = len(glob.glob(os.path.join(main_folder, "Orders/*")))

for i in range(n_orders):
# for i in [24]:
    order = i
    # print(order)
    data = np.loadtxt(os.path.join(main_folder, "Flux_raw/data_raw_order_{}.csv").format(order),  delimiter=',')
    cal_table_order = cal_lines[ cal_lines[:,0]==order+(reference_line_order-reference_order)]
    ThAr_data = gaussian_filter(data[0] - data[2], sigma = 5)
    x_pixels = np.arange(len(ThAr_data))
    # plt.plot(x_pixels, ThAr_data)
    # plt.show()
    x_peak = []
    labda = []
    peak_width = []
    for i in range(len(cal_table_order)):

        guess = int(cal_table_order[i][1])
        slice_width = 42
        slice_set_x = x_pixels[guess-slice_width:guess+slice_width]
        slice_set_y = ThAr_data[guess-slice_width:guess+slice_width]
        # print(order)
        # plt.plot(slice_set_x,slice_set_y)
        # plt.show()
        init_vals = [1, 0, 1]  # for [amp, cen, wid]
        best_vals, covar = curve_fit(gaussian, slice_set_x, slice_set_y, p0=[100000, guess, 10])
        
        peak_width.append(best_vals[2])
        
        
        x_peak.append(best_vals[1])
        labda.append(cal_table_order[i][3])
    
    FWHM=2.355*np.sqrt(np.median(peak_width)/2)
    
    # Polynomial fit for wavelength calibration
    fit = np.polynomial.polynomial.polyfit(x_peak,labda,3)
    # x & y coordinaten van de fit
    x_fit = []
    y_fit = []
    for x in range(len(ThAr_data)):
        y = 0
        # Calculate y_coordinate
        for n in range(len(fit)):
            y += fit[n] * (x)**n       
        # Save coordinates
        x_fit.append(x)
        y_fit.append(y)     
    
    # plt.figure(figsize = (20,16))
    # plt.title(f"order {order}")
    # plt.plot(x_fit,y_fit)
    # plt.scatter(x_peak, labda)
    # plt.show()
    
    # Make residual plot
    
    residuals = []
    for x_value in range(len(x_peak)):
        y=0
        for n in range(len(fit)):
            y += fit[n] * (x_peak[x_value])**n       
        residuals.append(labda[x_value] - y)
    plt.title('Residual plot')
    if order == 23:
        plt.plot(x_peak,residuals, linestyle='--', label='order{}'.format(order))
    else:
        plt.plot(x_peak,residuals, label='order{}'.format(order))

    print('dispersion (A/pix):', np.round(fit[1],4), 'in order', order, ',wavelength range (A):',round(np.min(y_fit),0),'-',round(np.max(y_fit),0), ',R=',np.round((np.min(y_fit)/2+np.max(y_fit)/2)/np.abs(FWHM*fit[1])) )
    data_cal = np.vstack((y_fit,data[0],data[1],data[2],data[3],data[4],data[5],data[6]))
    
    np.savetxt(os.path.join(main_folder, 'Flux_wavelength/Data_wavelength_cal_order{}.csv'.format(order)), data_cal,  delimiter=',')

plt.show()




# formatting of exported datasets: Wavelength, Thar,Tungsten,Bias,Dark,Object, SNR, darkflat
# %%

# _____________________________________________________
# __________________Automatic normalization____________
# _____________________________________________________

orders_list = glob.glob(os.path.join(main_folder, "Flux_wavelength/*"))
n_orders = len(orders_list)
# formatting of imported datasets: Wavelength, Thar,Tungsten,Bias, Dark,Object,SNR,  darkflat
for i in range(n_orders):
    data =  np.loadtxt(os.path.join(main_folder, f'Flux_wavelength/Data_wavelength_cal_order{i}.csv'),  delimiter=',')

    x=data[0]
    y=(data[5] - data[4]) / gaussian_filter(np.abs(data[2]-data[7])+1, sigma=5)
    # plt.plot(x,y)
    # plt.show()

    original_x = x
    original_y = y


    for j in range(20):
        
        fit = np.polynomial.polynomial.polyfit(x,y,20)
        
        y_fit = []
        for labda in range(len(x)):
            y_data = 0
            # Calculate y_coordinate
            for n in range(len(fit)):
                y_data += fit[n] * (x[labda])**n       
            # Save coordinates
            y_fit.append(y_data)  
        # plt.plot(x,y_fit, label = 'fit iteration{}'.format(j))
        y=np.asarray(y)
        residuals = y - y_fit 
        residuals_cut = np.std(residuals)
        x_new=[]
        y_new=[]
        for a in range(len(x)):
            if np.abs(residuals[a]) < 2*residuals_cut: 
                x_new.append(x[a])
                y_new.append(y[a])
        
        x=x_new
        y=y_new
        # print('stdev residuals', residuals_cut)
        
    #     plt.plot(x,y)
    #     plt.plot(x_new,y_new, label = 'iteration{}'.format(j) )
    
    # plt.legend()
    # plt.show()


    
    y_norm = []
    for labda in range(len(original_x)):
        y_data = 0
        # Calculate y_coordinate
        for n in range(len(fit)):
            y_data += fit[n] * (original_x[labda])**n       
        # Save coordinates
        y_norm.append(y_data)
    
    spectrum_norm = original_y/y_norm
    
    plt.figure(figsize = (8,6))
    plt.title(f'order {i}')
    plt.plot(original_x,spectrum_norm)
    plt.ylim(0,1.2)
    plt.show()

    data_sav = np.vstack((data[0],data[1],data[2],data[3],data[4],data[5],data[6],data[7],spectrum_norm))
    
    np.savetxt(os.path.join(main_folder, f'Flux_norm/Data_wavelength_norm_order{i}.csv'), data_sav,  delimiter=',')
# formatting of exported datasets: Wavelength, Thar,Tungsten,Bias, Dark,Object,SNR,  darkflat,object_norm

# %%

# _____________________________________________________
# __________________Merge* orders______________________
# _____________________________________________________

orders_list = glob.glob(os.path.join(main_folder, "Flux_norm/*"))

n_orders = len(orders_list)

full_data = []

# formatting of imported datasets: Wavelength, Thar,Tungsten,Bias, Dark,Object,SNR, darkflat,object_norm
for i in range(n_orders):
    data =  np.loadtxt(os.path.join(main_folder, f'Flux_norm/Data_wavelength_norm_order{i}.csv'),  delimiter=',')
    full_data.append(data)
    plt.plot(data[0],data[2], color='gray')
    
full_data_2 =[]
cut_off_2 = 10000
cut_off = 0    
offset = 5 # Angstrom shift
for i in range(n_orders): 
    
    if i == n_orders-1:
        
        data_set = full_data[i]
        begin_i = np.min(data_set[0])               
        data_add = data_set[:,(data_set[0] < cut_off-offset)]
        full_data_2.append(data_add)

        
    else:
        data_set = full_data[i]
        begin_i = np.min(data_set[0])
        end_iplus1 = np.max(full_data[i+1][0])
        cut_off=(begin_i/2+end_iplus1/2)
        
        data_add = data_set[:,(data_set[0] > cut_off-offset)]
        data_add = data_add[:,(data_add[0] < cut_off_2-offset)]
        full_data_2.append(data_add)
        cut_off_2 =cut_off
            
    plt.plot(data_add[0], data_add[2])
plt.show()

data_merged = full_data_2[0]
for i in range(1,len(full_data_2)):    
    data_merged = np.append(data_merged,full_data_2[i],axis =1)



# %%
# formatting of dataset: Wavelength, Thar,Tungsten,Bias, Dark,Object,SNR,  darkflat,object_norm

wavelength = data_merged[0]
thar = data_merged[1]
tungsten = data_merged[2]
bias = data_merged[3]
dark = data_merged[4]
obj = data_merged[5]
SNR = data_merged[6]
dark_flat = data_merged[7]
data_norm = data_merged[8]

flux_smooth= gaussian_filter(data_norm,sigma = 5)


# plt.plot(wavelength, thar, label = 'ThAr')
# plt.plot(wavelength, (obj-dark)/(tungsten-bias), label = 'Object')
# plt.plot(wavelength, tungsten, label = 'Tungsten')
# plt.plot(wavelength, dark, label = 'Dark')
# plt.plot(wavelength, bias, label = 'Bias')
plt.plot(wavelength, flux_smooth, label = 'SNR')
plt.show()

# plt.plot(wavelength, flux_smooth)
# plt.show()

np.savetxt(os.path.join(main_folder, 'Reduced/wave_flux.txt'), np.vstack((wavelength,flux_smooth)).T)




# %%
# use the brightest lines to find the shift in wavelength
# begin of list is red, end is blue
cutout_flux = np.copy(flux_smooth)
cutout_wl = np.copy(wavelength)

# only look at normalised signal around the known lines (4101.74 has a weird curve)
known_Fr_H_lines = np.sort([6562.8518,4861.3615,4340.462])
# Hydrogen alpha to delta
closest_to_indices = []
search_range = 10 # in A

plt.plot(cutout_wl, cutout_flux, label = 'P Cyg spectrum')
plt.hlines(y=1-np.std(cutout_flux),xmin=min(cutout_wl),xmax=max(cutout_wl), linestyle='--')
plt.vlines(x=known_Fr_H_lines,ymin=-0.5,ymax=1.5, linestyle='--', color='k')
plt.ylim(0,1.2)
plt.grid()
plt.show()


for i in range(len(known_Fr_H_lines)):
    line_found = False
    for j in range(len(cutout_wl)):
        if cutout_wl[j] <= known_Fr_H_lines[i] and line_found == False:
            closest_to_indices.append(j)
            line_found = True



def gaussian(x, amp, cen, wid, cont):
    return amp * np.exp(-(x-cen)**2 / wid) + cont

redshifts = []
redshift_err = []

range_index = 42
for i in range(len(closest_to_indices)):

    specific_cutout_flux = np.copy(cutout_flux[closest_to_indices[i]-range_index:closest_to_indices[i]+range_index])
    specific_cutout_wl = np.copy(cutout_wl[closest_to_indices[i]-range_index:closest_to_indices[i]+range_index])

    x_crop = specific_cutout_wl
    y_crop_object = specific_cutout_flux
    
    best_vals_object, covar = curve_fit(gaussian, x_crop, y_crop_object, p0=[-1, cutout_wl[closest_to_indices[i]], 1, np.min(y_crop_object)])
    error = np.sqrt(np.diag(covar))
    test_x = np.linspace(min(specific_cutout_wl), max(specific_cutout_wl), 1000)
    object_fit = gaussian(test_x, *best_vals_object)
    
    #'''
    plt.plot(wavelength, flux_smooth, label = 'VEGA spectrum')
    plt.vlines(x=known_Fr_H_lines[i],ymin=-0.5,ymax=1.5, linestyle='--', color='k')
    plt.vlines(x=best_vals_object[1],ymin=-0.5,ymax=1.5, linestyle='--', color='r')
    plt.plot(test_x, object_fit, color = "green", linewidth=2 , label = 'fit')
    plt.ylim(0,1)
    plt.xlim(cutout_wl[closest_to_indices[i]]-3,cutout_wl[closest_to_indices[i]]+3)
    plt.grid()
    plt.show()
    #'''
    
    lambda_observed = best_vals_object[1]
    lambda_emitted = known_Fr_H_lines[i]
    redshift = (lambda_observed - lambda_emitted)/(lambda_emitted)
    count_after_decimal = str(lambda_emitted)[::-1].find('.')
    count_after_decimal = 10**(-count_after_decimal)
    error_redshift = np.sqrt(error[1]**2 +count_after_decimal**2 + count_after_decimal**2)
    c = 299792458
    speed = redshift*c
    #'''
    print("known H line:", known_Fr_H_lines[i])
    print("fitted result:", round(best_vals_object[1],5), "±", round(error[1],5))
    print("redshift =", round(redshift,10))
    print("speed =", round(speed,10), "m/s")
    #'''
    
    redshifts.append(redshift)
    redshift_err.append(error_redshift)
    

# now determine the alleged 'real' redshift using a horizontal line fit
def line(x,b):
    return 0*x + b

best_vals_object, covar = curve_fit(line, known_Fr_H_lines, redshifts, p0=[1])
line_value = best_vals_object
line_error = np.sqrt(np.diag(covar))
error_plot = 0.1

plt.errorbar(known_Fr_H_lines, redshifts, yerr=np.multiply(redshift_err,error_plot), fmt="o", color='r', label=f'redshift with {error_plot} * error plotted')
plt.hlines(y=line_value[0],xmin=min(cutout_wl),xmax=max(cutout_wl), linestyle=':', label=f'linear redshift fit: {round(redshifts[i],10)}')
for i in range(len(known_Fr_H_lines)):
    plt.text(known_Fr_H_lines[i]+75, redshifts[i]-10**(-6), f'error: {round(redshift_err[i],5)}', size=10)
plt.xlabel("wavelength (A)")
plt.ylabel("redshift")
max_err = max(redshift_err) * error_plot
plt.ylim(line_value[0]-max_err/2,line_value[0]+max_err/2)
plt.legend(loc='lower right')
plt.grid()
plt.show()

print("fitted redshift result :", round(line_value[0],7), "±", round(line_error[0],10))
rv = line_value[0]*c
rv_err = (c*line_error[0])
print("radial velocity        :", round(rv,4), "±", round(rv_err,5), "m/s")















