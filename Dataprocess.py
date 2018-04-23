import h5py
import numpy
import math
from PIL import Image
import os

'''
this code can't compress images,
you must cmmpress image by matlab or other thing and put compressed images in a folder and labels in a floder
'''

root_dir=os.getcwd()
DATA_PATH = os.path.join(root_dir,'Images','data')
LABEL_PATH = os.path.join(root_dir,'Images','label')

patch_size=42
stride=16

def prepare_data(path):
    names = os.listdir(path)
    names = sorted(names)#Attention
    nums = names.__len__()

    data = []
    for i in range(nums):
        name=os.path.join(path,names[i])
        img = Image.open(name)
        #img =img.convert('YCbCr')#Attention
        img=numpy.asarray(img)
       
        shape=img.shape
        print 'img.shape:',shape
        row_num = ((shape[0] - patch_size) / stride)+1
        col_num =((shape[1] - patch_size) / stride)+1
        row_shift = (shape[0] - ((row_num - 1) * stride + patch_size)) / 2
        col_shift = (shape[1] - ((col_num - 1) * stride + patch_size)) / 2
        print 'row_num:',row_num,'col_num:',col_num
        
        for x in range(row_num):
            x_start = row_shift + (x) * stride
            x_end = row_shift + (x) * stride + patch_size
            for y in range(col_num):
                y_start = col_shift + (y) * stride
                y_end = col_shift + (y) * stride + patch_size
                sub_img = img[x_start:x_end, y_start:y_end, :]
                data.append(sub_img)

    data = numpy.array(data, dtype= numpy.float32)  #list has no shape
    print 'data.shape:',data.shape,data.dtype     
    return data

def write_hdf5(data, label,output_filename):
    """
    This function is used to save image data or its label(s) to hdf5 file.
    output_file.h5,contain data and label
    """
    x = data.astype(numpy.float32)
    y = label.astype(numpy.float32)
    with h5py.File(output_filename, 'w') as h:
        h.create_dataset('data', data=x, shape=x.shape)
        h.create_dataset('label', data=y, shape=y.shape)


if __name__ == "__main__":
    data= prepare_data(DATA_PATH)
    label = prepare_data(LABEL_PATH)
    #write_hdf5(data, label, '/home/test/wjq/Data10.h5')
    write_hdf5(data, label, os.path.join(root_dir,'Data10.h5'))
 

