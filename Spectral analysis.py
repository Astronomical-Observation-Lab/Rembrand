# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 12:04:26 2023

@author: Rasjied
"""
# from __future__ import print_function, division
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

# Load the vega spectrum

data_vega = np.loadtxt(os.path.join(main_folder, 'Reduced/wave_flux.txt'))
lines_removed = data_vega.copy()
only_peaks = data_vega.copy()
# data_vega[:,1]=gaussian_filter(data_vega[:,1],sigma =2)
# plt.plot(data_vega[:,0], data_vega[:,1])
# plt.ylim(0,1.2)
# plt.xlim(4200,7500)
# plt.show()

# just some fraunhofer lines that are also in the spectrum of vega

min_wl = min(data_vega[:,0]) # = 3831.5 A
max_wl = max(data_vega[:,0]) # = 7161.6 A
tot_range = max_wl - min_wl
tot_len = len(data_vega[:,0])

peak_wl = []
peak_int = []

split = 10
for i in range(split):
    if i < split:
        domain = tot_range / split
        min_value = int(i*(tot_len / split))
        max_value = int((i+1)*(tot_len / split))
        list_cutout = lines_removed[min_value:max_value]
        peak_cutout = only_peaks[min_value:max_value]
        #list_cutout = data_vega[min_value:max_value]
        std_value = list_cutout[:,1].std()
        for j in range(len(list_cutout[:,1])):
            value = list_cutout[j,1]
            if value < (1-std_value) or value > (1+std_value):
                list_cutout[j,1] = 1
        
        # reevaluate std
        std_value = list_cutout[:,1].std()
        
        for j in range(len(peak_cutout[:,1])):
            value = peak_cutout[j,1]
            if value < (1-2*std_value) or value > (1+2*std_value):
                peak_int.append(value)
                peak_wl.append(peak_cutout[j,0])
            else:
                peak_int.append(1)
                peak_wl.append(peak_cutout[j,0])
                
        
for i in range(split):
    if i < split:
        plt.plot(data_vega[:,0], data_vega[:,1])
        plt.scatter(peak_wl, peak_int, color = 'r', s=0.25)
        plt.ylim(0.6,1.4)
        plt.xlim(i*domain + min_wl,(i+1)*domain + min_wl)
        plt.show()

    
# %%       
# now we have a selection of all possible peaks, with zero+1 values between them
# we use this to average out the wavelength of each peak, and the cross with NIST data


# now to get a domain of values between zeroes, and get an average value
wl_values = []
wl_range_count = []
count = 0
non_zero = False
wl_additive = 0

print(len(peak_wl),len(peak_int))

for i in range(len(peak_int)):
    # peak ended or start
    if peak_int[i] == 1 and non_zero == True:
        # add the found wavelength to a list to use
        threshhold = 10
        if count > threshhold:
            wl = wl_additive / count
            wl_values.append(wl)
            wl_range_count.append(count)
        # reset for next non-zero value
        non_zero = False
        count = 0
        wl_additive = 0
    
    # peak found
    elif peak_int[i] != 1:
        non_zero = True
        count = count + 1
        wl_additive = wl_additive + peak_wl[i]

plt.scatter(wl_values, wl_range_count, color = 'r', s=0.5)
plt.show()

    
# %%
# now to check elements and see what lines roughly correspond

def ImportNISTData(File_Name):

    # create variables and lists
    row = 0
    Intensities = []
    Wavelengths = []
    Spectra = []
    Refs = []


    # open file
    input_file = open(f'{File_Name}.csv', 'r')

    # make lists of data
    for row_count in input_file:
        # skip first row (names etc.)
        if row > 0:
            
            # read out per line
            data_split = row_count.split(',')

            Intensity = str(data_split[0])
            Wavelength = float(data_split[1])
            Spectrum = str(data_split[2])
            Ref = str(data_split[3])

            # add to arrays
            Intensities.append(Intensity)
            Wavelengths.append(Wavelength)
            Spectra.append(Spectrum)
            Refs.append(Ref)

            
        row = row + 1

    input_file.close()

    #print("Data imported.")

    return Wavelengths

def CheckWavelengths(Filename):
    Emission_lines = ImportNISTData(Filename)
    correct_measurements = []
    for i in range(len(Emission_lines)):
        # we know the overall shift is around -8.9km/s,
        # so we know how much the lines should shift
        # look near this value
        redshift_speed = -8900
        c = 2.998*10**8
        delta_wl = (redshift_speed / c) * Emission_lines[i]
        supposed_wl = Emission_lines[i] + delta_wl
        for j in range(len(wl_values)):
            diff = 3
            if wl_values[j] > (supposed_wl - diff) and wl_values[j] < (supposed_wl + diff):
                correct_measurements.append(Emission_lines[i])
    return correct_measurements

# edit this to get data for different elements
# WARNING -- NOT PERFECT
# there are lines included where p cyg profiles are just not visible
# human control required

# fill in the lines where a clear p cyg profile can be seen
H_profile_lines = [4340.462,4861.2786,4861.287,6562.711,6562.7248,6562.8518]
He_profile_lines = [4387.929,4471.479,4471.68,4713.146,4713.38,4921.931,5015.678,5875.6148,5875.6404,5875.9663,6560.1,6678.1517,7065.1771,7065.2153,7065.7086]
Hg_profile_lines = [4339.223,5128.442,5677.105,5871.279,5888.939] # last line not sure what but there is something there
Fe_profile_lines = [3878.5732,3886.2822,3888.5134,3895.6565,3927.9199,3930.2966,4063.5942,4071.738,4415.1226,4427.2979,4920.5029,5167.4883] # same with most of these lines
Ba_profile_lines = [5997.087,6675.27,7059.943]
Em_lines = CheckWavelengths("Ba_lines")
print(Em_lines)
    
# %%
# ____________________________________________________
# ____________lowest value in line____________________
# ____________________________________________________


#lines_of_interest = [6562.8518,4861.3615,4340.462]
lines_of_interest = Em_lines

# lets start with reading the line center by taking the 'lowest value':
v_rad_list = []
    
for i in lines_of_interest:
    slice_width = 150 # datapoints
    center_index = (np.abs(data_vega[:,0] - i)).argmin() 
    data_slice = data_vega[center_index-slice_width:center_index+slice_width]
    data_slice_x = data_slice[:,0]
    data_slice_y = data_slice[:,1]
    lowest_value_y = np.min(data_slice_y)
    lowest_value_x = np.argwhere(data_slice_y==lowest_value_y)
    center_wavelength = data_slice_x[lowest_value_x]
    v_rad = 300000*(center_wavelength - i)/ i #km/s
    v_rad = np.round(v_rad,2)
    v_rad_list.append(v_rad)
    print('v_rad:', v_rad, 'km/s.')  
    plt.title(f'line at around {i} Angstrom')
    plt.plot(data_slice_x, data_slice_y)
    plt.scatter(center_wavelength,lowest_value_y, color = 'red')
    plt.axvline(i, linestyle = '--', color = 'gray')
    plt.show()

print('lowest value method: mean v_rad',np.round(np.mean(v_rad_list)), 'km/s. StDev:', np.round(np.std(v_rad_list)), 'km/s')

# %%
# ____________________________________________________
# ____________Gaussian fit____________________________
# ____________________________________________________


def gaussian(x, amp, cen, wid, const):
    return amp * np.exp(-(x-cen)**2 / wid) + const

def doubleGaussian(x, amp1, amp2, cen1, cen2, wid1, wid2, const1, const2):
    gauss1 = amp1 * np.exp(-(x-cen1)**2 / wid1) + const1
    gauss2 = amp2 * np.exp(-(x-cen2)**2 / wid2) + const2
    return gauss1 + gauss2

v_rad_list_gaussian = []
v_rad_c_list_gaussian = []
v_rad_errors = []
v_rad_c_errors = []

for i in lines_of_interest:
    center_index = (np.abs(data_vega[:,0] - i)).argmin() 
    data_slice = data_vega[center_index-slice_width:center_index+slice_width]
    
    max_shift_peak_index = np.where(data_slice[:,1] == max(data_slice[:,1]))[0][0]
    min_shift_peak_index = np.where(data_slice[:,1] == min(data_slice[:,1]))[0][0]
    max_shift_wl = data_slice[:,0][max_shift_peak_index]
    min_shift_wl = data_slice[:,0][min_shift_peak_index]
    
    double_init_vals = [-100, 100, min_shift_wl, max_shift_wl, 0.2, 0.2, 1, 1]
    best_vals, covar = curve_fit(doubleGaussian, data_slice[:,0], data_slice[:,1], p0=double_init_vals)
    print(best_vals)
    y_vals = doubleGaussian(data_slice[:,0],*best_vals)
    
    # remove blueshifted gaussian peak
    corrected_gaussian = []
    for index in range(len(best_vals)):
        if index % 2 == 0 or index == 7:
            corrected_gaussian.append(best_vals[index])
        else:
            corrected_gaussian.append(0)
    y_corr_vals = np.subtract(doubleGaussian(data_slice[:,0],*corrected_gaussian),1)
    corr_data = np.subtract(data_slice[:,1],y_corr_vals)
    
    plt.plot(data_slice[:,0], doubleGaussian(data_slice[:,0],*best_vals), label='dubble Gaussian fit', color='b')
    plt.plot(data_slice[:,0], data_slice[:,1], label='eShel data', color='r')
    plt.plot(data_slice[:,0], corr_data, label='corrected data', color='g')
    plt.axvline(i, linestyle = '--', label='known Fraunhofer line', color = 'k')
    plt.axhline(1, linestyle = '--', label='intensity = 1', color = 'k')
    # plt.ylim(0,1.2)
    plt.legend()
    plt.show()
    center_wavelength = best_vals[3]
    v_rad = 300000*(center_wavelength - i)/ i #km/s
    v_rad_list_gaussian.append(v_rad)
    wavelength_error = covar[1][1]
    v_rad_error = np.round( v_rad - (300000*(center_wavelength+wavelength_error - i)/ i)  ,4)
    v_rad_errors.append(v_rad_error)
    print('v_rad:', round(v_rad,2), 'km/s. (Fit error:',v_rad_error, ').')
    
    # now fit a single gaussian on this corrected data and determine the true shift
    peak_index = np.where(corr_data == max(corr_data))[0][0]
    peak_wl = data_slice[:,0][peak_index]
    init_vals = [100, peak_wl, 0.2, 1]
    best_vals, covar = curve_fit(gaussian, data_slice[:,0], corr_data, p0=init_vals)
    plt.plot(data_slice[:,0], gaussian(data_slice[:,0],*best_vals), label='Gaussian fit', color='b')
    plt.plot(data_slice[:,0], corr_data, label='corrected data', color='g')
    plt.axvline(i, linestyle = '--', label='known Fraunhofer line', color = 'k')
    plt.axhline(1, linestyle = '--', label='intensity = 1', color = 'k')
    plt.legend()
    plt.show()
    print(best_vals)
    center_wavelength = best_vals[1]
    v_rad_c = 300000*(center_wavelength - i)/ i #km/s
    v_rad_c_list_gaussian.append(v_rad_c)
    wavelength_c_error = covar[1][1]
    v_rad_c_error = np.round( v_rad_c - (300000*(center_wavelength+wavelength_c_error - i)/ i)  ,4)
    v_rad_c_errors.append(v_rad_c_error)
    print('v_rad:', round(v_rad_c,2), 'km/s. (Fit error:',v_rad_c_error, ').')
    
    
    if i == lines_of_interest[0]:
        H_a_profile_norm = np.divide(np.subtract(data_slice[:,1],1),max(np.subtract(data_slice[:,1],1)))
        H_a_corr_norm = np.divide(np.subtract(corr_data,1),max(np.subtract(corr_data,1)))
        v_rad_Ha = v_rad
        v_rad_Ha_c = v_rad_c
    elif i == lines_of_interest[1]:
        H_b_profile_norm = np.divide(np.subtract(data_slice[:,1],1),max(np.subtract(data_slice[:,1],1)))
        H_b_corr_norm = np.divide(np.subtract(corr_data,1),max(np.subtract(corr_data,1)))
        v_rad_Hb = v_rad
        v_rad_Hb_c = v_rad_c
    elif i == lines_of_interest[2]:
        H_g_profile_norm = np.divide(np.subtract(data_slice[:,1],1),max(np.subtract(data_slice[:,1],1)))
        H_g_corr_norm = np.divide(np.subtract(corr_data,1),max(np.subtract(corr_data,1)))
        v_rad_Hg = v_rad
        v_rad_Hg_c = v_rad_c
        
    x_vals = []
    for j in range(len(data_slice[:,0])):
        x_vals.append(len(data_slice[:,0])-j)
        
        
plt.title('normalised data')
plt.plot(x_vals, H_a_profile_norm, label=f'H alpha data, v_rad = {v_rad_Ha}km/s', color='red')
plt.plot(x_vals, H_b_profile_norm, label=f'H alpha data, v_rad = {v_rad_Hb}km/s', color='cyan')
plt.plot(x_vals, H_g_profile_norm, label=f'H alpha data, v_rad = {v_rad_Hg}km/s', color='blue')
plt.legend(prop={'size': 8}, loc='lower right')
plt.show()

plt.title('normalised corrected data')
plt.plot(x_vals, H_a_corr_norm, label=f'H alpha data, v_rad = {v_rad_Ha_c}km/s', color='red')
plt.plot(x_vals, H_b_corr_norm, label=f'H alpha data, v_rad = {v_rad_Hb_c}km/s', color='cyan')
plt.plot(x_vals, H_g_corr_norm, label=f'H alpha data, v_rad = {v_rad_Hg_c}km/s', color='blue')
plt.legend(prop={'size': 8}, loc='lower right')
plt.show()
    
print('doubleGaussian method: mean v_rad',np.round(np.mean(v_rad_list_gaussian),3), 'km/s. StDev:', np.round(np.std(v_rad_list_gaussian),3), 'km/s.')
print('corrected Gaussian method: mean v_rad_c',np.round(np.mean(v_rad_c_list_gaussian),3), 'km/s. StDev:', np.round(np.std(v_rad_c_list_gaussian),3), 'km/s.')

 
    
    
# %%  
# ____________________________________________________
# ____________Barycentric correction__________________
# ____________________________________________________

    
# install pyastronomy with pip install first


from PyAstronomy import pyasl

# Coordinates of Anton Pannekoek Observatory

latitude = 52.354332832031325
longitude=  4.955059519775514
altitude = 0

# Coordinates of vega (J2000)
ra2000 = 279.2333333
dec2000 = 38.7836111

# (Mid-)Time of observation

object_list = glob.glob(os.path.join(main_folder, "Object/*.fit"))
jd= fits.getheader(object_list[0])['JD']

# Calculate barycentric correction (debug=True show
# various intermediate results)
corr, hjd = pyasl.helcorr(longitude, latitude, altitude, \
            ra2000, dec2000, jd, debug=True)

# Note Positive return values indicate that the Earth moves toward the star.
    
print("Barycentric correction [km/s]: ", corr)
print("Heliocentric Julian day: ", hjd)










