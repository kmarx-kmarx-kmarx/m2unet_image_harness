from utils import *
import numpy as np
import time
from tqdm import tqdm
from tqdm.contrib.itertools import product
from run_on_image import inference_on_image_stack
import json
import logging
import csv


# Run m2unet on local data and save the binary masks locally in .npz format
# in the data_path folder, we expect to see a folder for each dataset containing the raw data
def main():
    debug = True
    logging.basicConfig(filename='timing.log', level=logging.DEBUG)
    data_path = "/media/prakashlab/Extreme SSD/octopi 2023/raw data"
    save_path = "/media/prakashlab/Extreme SSD/octopi 2023/masks32"
    model_path = "/home/prakashlab/Documents/kmarx/train_m2unet_cellpose_cloud/m2unet_model_8"
    model_name = "model_70_13.pth"
    dataset_file = 'local_datasets.txt'
    n_batch = 75
    thresh = 128
    # illumination correction
    flatfield_left = np.load('flatfield_left.npy')
    flatfield_right = np.load('flatfield_right.npy')

    with open(dataset_file,'r') as f:
        DATASET_ID = f.read()
        DATASET_ID = DATASET_ID.split('\n')[0:-1]
    if debug:
        try:
            DATASET_ID = DATASET_ID[0:2]
        except:
            pass
    
    for dataset in tqdm(DATASET_ID):
        t0 = time.time()
        # For each dataset, generate DPCs and load all the data into a numpy array and run M2Unet
        param_path = os.path.join(data_path, dataset, 'acquisition parameters.json')
        with open(param_path, 'r') as f:
            json_file = f.read()
        acquisition_parameters = json.loads(json_file)
        if debug:
            acquisition_parameters['Ny'] = 3
            acquisition_parameters['Nx'] = 3
        n_images = acquisition_parameters['Ny'] * acquisition_parameters['Nx']
        # Get image size
        dims = np.array(cv2.imread(os.path.join(data_path, dataset, '0', "0_0_0_BF_LED_matrix_right_half.bmp"))).shape
        width = dims[0]
        height = dims[1]

        dpc_array = np.zeros((n_images, width, height), dtype=np.uint8)
        for i, c in enumerate(product(range(acquisition_parameters['Nx']), range(acquisition_parameters['Ny']))):
            x, y = c
            # Load the images and generate a DPC
            file_id = f"{y}_{x}_0"
            dpc = get_dpc(data_path, dataset, file_id, flatfield_left, flatfield_right)
            dpc_array[i, :, :] = dpc
        dt = time.time() - t0
        t1 = time.time()
        logging.debug(f"Took {dt} seconds to load {n_images} images from {dataset}")
        # Batch segment the dpc_array. Break into smaller chunks if necessary
        result = np.zeros((n_images, width, height), dtype=bool)
        index = 0
        while index < n_images:
            end_idx = min(index + n_batch, n_images)
            result[index:end_idx, :, :], __ = inference_on_image_stack(dpc_array[index:end_idx, :, :], model_path, model_name, threshold_set=thresh)
            index = end_idx
        dt = time.time() - t1
        t2 = time.time()
        logging.debug(f"Took {dt} seconds to segment {n_images} images")

        # save the outputs
        savepath = os.path.join(save_path, dataset)
        os.makedirs(savepath, exist_ok=True)
        for i, c in enumerate(product(range(acquisition_parameters['Nx']), range(acquisition_parameters['Ny']))):
            x, y = c
            # Load the images and generate a DPC
            file_id = f"{y}_{x}_0_mask.npz"
            fname = os.path.join(savepath, file_id)
            np.savez(fname, mask=result[i, :, :])
        dt = time.time() - t2
        logging.debug(f"Took {dt} seconds to save {n_images} npz files")
        logging.debug(f"Total time for {dataset}: {time.time()-t0}\n")

if __name__ == "__main__":
    main()