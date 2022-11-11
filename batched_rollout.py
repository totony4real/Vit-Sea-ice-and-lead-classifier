#%%
import os
import netCDF4
import numpy as np
import xarray as xr
import pyproj
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
from sklearn.feature_extraction import image
from typing import Tuple, Union, Dict
from tqdm import tqdm


def WGS84toEASE2(lon, lat):
    """
    Convert from WGS84 to EASE2
    """
    proj_EASE2 = pyproj.Proj("+proj=laea +lon_0=0 +lat_0=90 +x_0=0 +y_0=0 +ellps=WGS84 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs")
    proj_WGS84 = pyproj.Proj("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs ")
    x , y = pyproj.transform(proj_WGS84, proj_EASE2, lon, lat)
    return x, y


class BatchedRollout:
    def __init__(self, OLCI_path, model, logdir='./log'):
        """
        Module to run batched-rollout on a given OLCI image
        -------
        Args:
        -------
        :OLCI_path: directory where OLCI image/metadata is stored
        :model:     vision transformer model
        :logdir:    target directory to save results
        """

        self.model = model
        self.input_shape = model.layers[0].output_shape[0][1:]
        self.logdir = logdir

        # Load in geolocation (use xarray)
        geolocation = netCDF4.Dataset(OLCI_path+'/geo_coordinates.nc')
        lat = geolocation.variables['latitude'][:]
        lon = geolocation.variables['longitude'][:]
        X, Y = WGS84toEASE2(lon, lat)
        self.image_height = X.shape[0]
        self.image_width = Y.shape[1]

        # Load radiance dataset
        self.radiance_dataset = xr.open_mfdataset(OLCI_path+'/*_radiance.nc', parallel=True)

        # Load instrument data
        self.instrument_dataset = xr.open_dataset(OLCI_path+'/instrument_data.nc')
        self.solar_flux = self.instrument_dataset['solar_flux'].values.T # Shape (3700, 21)
        self.detector_index = self.instrument_dataset['detector_index']

        # Load tie geometries
        self.tie_geometries = xr.open_dataset(OLCI_path+'/tie_geometries.nc')
        SZA = self.tie_geometries['SZA']
        angle = np.zeros((self.image_height, self.image_width, 1))
        for j in range(self.image_width):
            angle[:,j,0]=SZA[:,j//64]
        self.angle = angle

        # Get fill-in values to replace NaNs for later use (couldn't get with xarray)
        self.fill_value = {}
        self.fill_value['radiance'] = netCDF4.Dataset(OLCI_path+'/Oa01_radiance.nc').variables['Oa01_radiance']._FillValue
        self.fill_value['detector_index'] = netCDF4.Dataset(OLCI_path+'/instrument_data.nc').variables['detector_index']._FillValue

    def _extract_data_on_subregion(self, data, subregion: Tuple[int, int, int, int], fillvalue=0.0):
        """
        Get data on a selected subregion without loading the entire data onto memory
        """
        X, Y, H, W = subregion
        if data.__class__ == np.ndarray:
            subdata = data[X:X+H, Y:Y+W]
        else:
            subdata = data.sel(rows=slice(X,X+H), columns=slice(Y,Y+W))
            if subdata.__class__ == xr.Dataset:
                subdata = subdata.to_stacked_array("channel", sample_dims=["rows", "columns"], name='patch').values
        # Replace NaNs with fill-values
        subdata = np.nan_to_num(subdata, nan=fillvalue)
        return subdata

    def _compute_TOA_BRF(self, oa, detector_index, angle):
        """
        Compute the top-of-atmsophere bi-directional reflectance (TOA_BRF)
        """
        TOA_BRF = np.pi*oa / self.solar_flux[detector_index] / np.cos(np.radians(angle))
        return TOA_BRF

    def subregion_rollout(self,
                          subregion: Tuple[int, int, int, int],
                          minibatchsize: Union[int, None]
                          ):
        """
        Perform rollout on selected subregion
        Subregion is determined by a tuple (X, Y, H, W) where
        - (X, Y) is the top-left corner of the subregion, and
        - (H, W) is the height/width respectively of the subregion
        """

        X, Y, H, W = subregion
        inH, inW, _ = self.input_shape
        padded_subregion = (X, Y, H+(inH-1), W+(inW-1)) # Pad the subregion by an appropriate amount to account for edges

        # Compute top-of-atmosphere bi-directional reflectance on subregion
        oa = self._extract_data_on_subregion(self.radiance_dataset, padded_subregion, self.fill_value['radiance'])
        detector_index = self._extract_data_on_subregion(self.detector_index, padded_subregion, self.fill_value['detector_index'])
        detector_index = detector_index.astype(int)
        region_angle = self._extract_data_on_subregion(self.angle, padded_subregion)
        TOA_BRF = self._compute_TOA_BRF(oa, detector_index, region_angle) # Shape (H+(inH-1), W+(inW-1), inH, inW, C)

        x_test_subregion_full = image.extract_patches_2d(TOA_BRF, (inH, inW)) # Shape (H x W, inH, inW, C)

        N = H * W
        if minibatchsize == None:
            # Do a full-batch rollout if minibatchsize is not specified
            ypred = self.model.predict(x_test_subregion_full) # Shape (N, inH, inW, C)
        else:
            # Do mini-batch rollout otherwise
            dataloader = tf.data.Dataset.from_tensor_slices(x_test_subregion_full)
            num_batches = np.ceil(N / minibatchsize).astype(int)
            ypred = []
            # for i, batch in enumerate(dataloader.batch(minibatchsize)):
            for batch in tqdm(dataloader.batch(minibatchsize)):
                # print(f"Batch {i+1}/{num_batches}")
                y = self.model.predict(batch, verbose=0) # Shape (M, inH, inW, C)
                ypred.append(y)
            ypred = np.concatenate(ypred)

        ypred = np.argmax(ypred, axis=1).reshape(H, W) # Get 0/1 predictions
        return ypred


    def full_rollout(self,
                subregion_shape=(100, 100),
                minibatchsize=None,
                save=True
                ):
        """
        Main loop to perform rollout on the entire OLCI image.
        ------------
        Example:
        ------------
        If

        OLCI image size = (500, 500), and
        subregion size = (100, 100)

        then the OLCI image will first be divided into (500/100, 500/100) = (5, 5) subregions.
        Rollout will be performed on each subregion one at a time.

        Further, let the input shape of the model be

        input_shape = (3, 3, 21)

        Then on each subregion (i, j) for i, j = 1,...,5, we feed the model an input tensor X of size (N, H, W, C) = (100*100, 3, 3, 21),
        which outputs a tensor y of size (N,) = (100*100,)

        We can also minibatch the input tensor X to fit into GPU memory during rollout.
        E.g. if minibatch_size = 100, then it will perform 100 iterations of rollout per subregion, where at each iteration,
        an input tensor of size (M, H, W, C) = (100, 3, 3, 21) is fed into the model.

        Tips:
        ---------
        - If system memory is exhausted -> use smaller subregion size
        - If GPU memory is exhausted -> use smaller minibatch size
        """

        inH, inW, _ = self.input_shape # Get directly from model?

        # Compute effective image height and width. A little bit is taken off to account for edge cases.
        HEIGHT = self.image_height-(inH-1)
        WIDTH = self.image_width-(inW-1)

        # Number of subregions (including/excluding remainder regions)
        num_rows_including_remainders = np.ceil(HEIGHT / subregion_shape[0]).astype(int)
        num_cols_including_remainders = np.ceil(WIDTH / subregion_shape[1]).astype(int)
        num_rows_excluding_remainders = HEIGHT // subregion_shape[0]
        num_cols_excluding_remainders = WIDTH // subregion_shape[1]
        remainder_height = HEIGHT % subregion_shape[0]
        remainder_width = WIDTH % subregion_shape[1]
        num_subregions = num_rows_including_remainders * num_cols_including_remainders

        print(f"Total number of subregions: {num_subregions}")
        if minibatchsize == None:
            print("Full-batch rollout...")
        else:
            print(f"Mini-batch rollout with batchsize={minibatchsize}")

        count = 1
        chunked_outputs = {}
        for i in range(num_rows_including_remainders):
            for j in range(num_cols_including_remainders):
                print(f"Subregion: {count}/{num_subregions}")

                # Extract subregion
                X = i * subregion_shape[0]
                Y = j * subregion_shape[1]
                H = subregion_shape[0] if i != num_rows_excluding_remainders else remainder_height
                W = subregion_shape[1] if j != num_cols_excluding_remainders else remainder_width
                
                # Perform rollout on selected subregion
                subregion = (X, Y, H, W)
                ypred = self.subregion_rollout(subregion=subregion, minibatchsize=minibatchsize)

                # Save result
                chunked_outputs[(i, j)] = ypred
                if save == True:
                    np.savez(self.logdir+f'/subregion_{count}.npz', key=np.array((i,j)), data=ypred)

                count += 1

        full_prediction = self.combine_predictions(chunked_outputs)

        print("Complete!")
        
        return full_prediction

    @staticmethod
    def combine_predictions(chunked_outputs: Dict):
        """
        Combine predictions on each subregion to get a single combined prediction on the full image
        ------
        Args:
        ------
        :chunked_outputs: A dictionary of form {'subregion key': prediction at subregion} where the
                          subregion key (i, j) indicates the relative position of the block within
                          the whole image. E.g. if the whole image is divided into 2x2 subregions,
                          then the subregion keys will consist of (0,0), (0,1), (1,0), (1,1)
        """
        subregion_idxs = np.array(list(chunked_outputs.keys())) # Indices of subregion (encodes relative positions of the blocks)
        num_rows, num_cols = np.max(subregion_idxs, axis=0)
        num_rows += 1 # Account for the fact that the first index is 0
        num_cols += 1
        blocked_predictions = [[chunked_outputs[(i,j)] for j in range(num_cols)] for i in range(num_rows)]
        combined_prediction = np.block(blocked_predictions)
        return combined_prediction

#%%
if __name__ == "__main__":
    # Set OLCI directory path
    path = '/home/so/Documents/Projects/Vit-Sea-ice-and-lead-classifier/data/'
    directory = 'S3B_OL_1_EFR____20190301T232521_20190301T232821_20200111T235148_0179_022_301_1800_MR1_R_NT_002.SEN3'
    OLCI_path = path + directory

    # Load ViT model
    model = keras.models.load_model('/home/so/Documents/Projects/Vit-Sea-ice-and-lead-classifier/Pre_trained_model')

    # Set up rollout module
    f = BatchedRollout(OLCI_path, model)

    #%%
    # Perform test rollout on a selected subregion
    X = 1000
    Y = 1000
    H = 400
    W = 400
    subregion = (X, Y, H, W)
    minibatchsize = 256
    
    ypred_subregion = f.subregion_rollout(subregion=subregion, minibatchsize=minibatchsize)

    plt.imshow(ypred_subregion)
    plt.title("predictions on a subregion")
    plt.show()

    # %%
    # Perform rollout on the full OLCI image
    subregion_shape = (400, 400)
    minibatchsize = 256
 
    _ = f.full_rollout(subregion_shape=subregion_shape, minibatchsize=minibatchsize, save=True) # May take hours to complete

    # %%
    # Plot predictions on the full image
    from_saved = True

    if from_saved == True:
        f = BatchedRollout(OLCI_path, model)
        inH, inW, _ = f.input_shape
        HEIGHT = f.image_height
        WIDTH = f.image_width
        HEIGHT -= (inH - 1)
        WIDTH -= (inW - 1)
        num_rows = np.ceil(HEIGHT / subregion_shape[0]).astype(int)
        num_cols = np.ceil(WIDTH / subregion_shape[1]).astype(int)
        num_subregions = num_rows * num_cols

        chunked_outputs = {}
        for i in range(1,num_subregions+1):
            loadfile = np.load(f'log/subregion_{i}.npz')
            key = loadfile['key']
            pred = loadfile['data']
            chunked_outputs[tuple(key)] = pred

        full_prediction = BatchedRollout.combine_predictions(chunked_outputs)
        plt.imshow(full_prediction)
        plt.title('predictions on the full image')

    else:
        plt.imshow(_)
        plt.title('predictions on the full image')


# %%
