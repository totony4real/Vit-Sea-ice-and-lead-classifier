#pip install pyproj
#pip install cartopy


models = keras.models.load_model('')
path = ''

import os
import netCDF4
import numpy as np 
import pyproj
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from math import pi
from sklearn.datasets import load_sample_image
from sklearn.feature_extraction import image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 600


def WGS84toEASE2(lon, lat):
    proj_EASE2 = pyproj.Proj("+proj=laea +lon_0=0 +lat_0=90 +x_0=0 +y_0=0 +ellps=WGS84 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs")
    proj_WGS84 = pyproj.Proj("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs ")
    x , y = pyproj.transform(proj_WGS84, proj_EASE2, lon, lat)
    return x, y

X, Y = WGS84toEASE2(lon, lat)

# Specify your directory = ''
directory = ''
# Load in geolocation
geolocation = netCDF4.Dataset(path+directory+'/geo_coordinates.nc')
lat = geolocation.variables['latitude'][:]
lon = geolocation.variables['longitude'][:]

# Load in radiance
Band_Oa01 = netCDF4.Dataset(path+directory+'/Oa01_radiance.nc')
Oa01_Radiance = Band_Oa01.variables['Oa01_radiance'][:]

OLCI_file_p=path+directory
instrument_data = netCDF4.Dataset(OLCI_file_p+'/instrument_data.nc')
solar_flux = instrument_data.variables['solar_flux'][:]
solar_flux_Band_Oa01 = solar_flux[0] # Band 1 has index 0 ect. 
detector_index = instrument_data.variables['detector_index'][:]

# Load in tie geometries
tie_geometries = netCDF4.Dataset(OLCI_file_p+'/tie_geometries.nc')
SZA = tie_geometries.variables['SZA'][:]

##Split images according to patches for prediction (with the gradient removed)
Bands=[]
Patches=[]
nx=X.shape[0]-2
ny=X.shape[1]-2
q = 0
for i in range(1,22):
    solar_flux_Band_Oa01 = solar_flux[q]
    print(i)
    bandnumber = '%02d' % (i)
    Band_Oa_temp = netCDF4.Dataset(path+directory+'/Oa'+bandnumber+'_radiance.nc')

    width = instrument_data.dimensions['columns'].size
    height = instrument_data.dimensions['rows'].size

    TOA_BRF = np.zeros((height, width), dtype='float32')
    angle=np.zeros((TOA_BRF.shape[0],TOA_BRF.shape[1]))
    for x in range(TOA_BRF.shape[1]):
      angle[:,x]=SZA[:,int(x/64)]

    width = instrument_data.dimensions['columns'].size
    height = instrument_data.dimensions['rows'].size

    #TOA_BRF = np.zeros((height, width), dtype=float)
    #TOA_BRF=np.pi*Oa01_Radiance/solar_flux_Band_Oa01[detector_index]/np.cos(np.radians(angle))
    oa = Band_Oa_temp.variables['Oa'+bandnumber+'_radiance'][:]
    TOA_BRF = np.zeros((height, width), dtype=float)
    TOA_BRF=np.pi*np.asarray(oa)/solar_flux_Band_Oa01[detector_index]/np.cos(np.radians(angle))
    Bands.append(TOA_BRF)
    Patches.append(image.extract_patches_2d(np.array(TOA_BRF), (3, 3)).reshape(nx,ny,3,3))
    q = q + 1

    Pathches_array=np.asarray(Patches)
Pathches_array.shape
x_test_all=np.moveaxis(Pathches_array,0,-1).reshape(Pathches_array.shape[1]*Pathches_array.shape[2],3,3,21)


#Start the prediction
y_pred=model.predict(x_test_all)
print(y_pred)
y_pred1 = np.argmax(y_pred,axis = 1)
print(np.argmax(y_pred,axis = 1))
Map=y_pred1.reshape(Pathches_array.shape[1],Pathches_array.shape[2])

