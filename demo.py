import cv2
import numpy as np
from run_on_image import inference_on_image_stack
import os

data_dir = "path/to/dpc/images"
ff_dir = "path/to/flatfield/npy"
model_root = "/path/to/model"
model_name = "modelname.pth"

def generate_dpc(I_1,I_2):
    I_dpc = np.divide(I_1-I_2,I_1+I_2)
    I_dpc = I_dpc + 0.5
    I_dpc[I_dpc<0] = 0
    I_dpc[I_dpc>1] = 1

    I_dpc = (255*I_dpc)

    return I_dpc.astype('uint8')

lft = cv2.imread(os.path.join(data_dir, '0_1_0_BF_LED_matrix_left_half.bmp'))[:,:,0]
rht = cv2.imread(os.path.join(data_dir, '0_1_0_BF_LED_matrix_right_half.bmp'))[:,:,0]
flatfield_left = np.load(os.path.join(ff_dir, 'flatfield_left.npy'))
flatfield_right = np.load(os.path.join(ff_dir, 'flatfield_right.npy'))
lft = lft.astype('float')/255
rht = rht.astype('float')/255

img = generate_dpc(lft, rht)

img = np.expand_dims(img, axis=0)

result = inference_on_image_stack(img, model_root, model_name, overlap=18)

cv2.imwrite("test.bmp", result[0,:,:])