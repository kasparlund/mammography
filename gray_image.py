from fastai.vision import *
import numpy as np
import PIL

#Function to open 16bit grayscale (fx an xray as png) and convert it to an rgb-tensor without loss of precision
#then you can assing it to the datasets like so: dsTrain.image_opener = dsValid.image_opener = open_image_16bit2rgb
# or use the datablock logic (i have not tried that)
def open_image_16bit2rgb( fn ): 
    a = np.asarray(PIL.Image.open( fn ))
    a = np.expand_dims(a,axis=2)
    a = np.repeat(a, 3, axis=2)
    return Image( pil2tensor(a, np.float32 ).div(65535) )

#Function to open 16bit grayscale (fx an xray as png) and convert it to an rgb-tensor without loss of precision
#then you can assing it to the datasets like so: dsTrain.image_opener = dsValid.image_opener = open_image_16bit2rgb
# or use the datablock logic (i have not tried that)
def open_image_rgb2rgbgray( fn ): 
    a = np.asarray( PIL.Image.open( fn ).convert("I"))
    a = np.expand_dims(a,axis=2)
    a = np.repeat(a, 3, axis=2)
    return Image( pil2tensor(a, np.float32 ).div(255) )    

def open_image_rgb_2_gray8bit( fn ): 
    t = pil2tensor( PIL.Image.open(fn).convert("I"), np.float32 ).div(255)
    return Image( t )
