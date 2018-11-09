import numpy as np
import os
import cv2

def RGB2GRAY( im ):
    #im = 0.299*R + 0.587*G + 0.114*B
    return 0.299*im[:,:,0] + 0.587*im[:,:,1] + 0.114*im[:,:,2]
def open_image_16bit2rgb( fn ): 
    a = np.asarray(PIL.Image.open( fn ).convert("I"))
    #print(a.shape)
    print(f"max image value {np.max(a)}")
    a = np.expand_dims(a,axis=2)
    a = np.repeat(a, 3, axis=2)
    #print(a.shape)
    return Image( pil2tensor(a, np.float32 ).div_(65535) )


class ImageReader:
    #outChannels the number of channels that must be returned from open
    #ie if the image is rgb then it is converted to graysale
    def __init__(self, outChannels=3):
        self.outChannels = outChannels

    def open(self, fn):
        """ Opens an image using OpenCV given the file path.

        Arguments:
            fn: the file path of the image

        Returns:
            The image in RGB og Grayscale format as numpy array of floats 
            normalized to range between 0.0 - 1.0
        """
        flags = cv2.IMREAD_UNCHANGED+cv2.IMREAD_ANYDEPTH+cv2.IMREAD_ANYCOLOR
        if not os.path.exists(fn) and not str(fn).startswith("http"):
            raise OSError('No such file or directory: {}'.format(fn))
        elif os.path.isdir(fn) and not str(fn).startswith("http"):
            raise OSError('Is a directory: {}'.format(fn))
        else:
            #res = np.array(Image.open(fn), dtype=np.float32)/255
            #if len(res.shape)==2: res = np.repeat(res[...,None],3,2)
            #return res
            try:
                if str(fn).startswith("http"):
                    req = urllib.urlopen(str(fn))
                    image = np.asarray(bytearray(req.read()), dtype="uint8")
                    im = cv2.imdecode(image, flags).astype(np.float32)/255
                else:
                    im = cv2.imread(str(fn), flags).astype(np.float32)/255
                if im is None: raise OSError(f'File not recognized by opencv: {fn}')

                if self.outChannels==3:
                    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) 
                elif len(im.shape)==3 and im.shape[2] == 3:
                    #Y = 0.299 R + 0.587 G + 0.114 B
                    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY )
                    im = np.expand_dims(im,2) / im.flatten().max()
                    
                    
                return im
            except Exception as e:
                raise OSError('Error handling image at: {}'.format(fn)) from e
