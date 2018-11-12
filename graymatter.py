from fastai import *
from fastai.vision import *
#from fastai.core import *
import numpy as np
#from torch import nn

#functions to convert the inputlayer of the nn architecture to grayscale input

def getGrayStats( imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) ):
    stats = imagenet_stats
    s=np.asarray(stats)
    st = []
    if len(s.shape)>=2 and s.shape[1] > 1:
        st.append( torch.from_numpy( np.asarray( np.mean(s[0]) ) ).float() )
        st.append( torch.from_numpy( np.asarray( np.sqrt( sum(s[1]*s[1]) / s.shape[1] ) ) ).float() ) 
    return st

def set_trainable(l, b):apply_leaf(l, lambda m: set_trainable_attr(m,b))

def set_trainable_attr(m,b):
    m.trainable=b
    for p in m.parameters(): p.requires_grad=b
        
def inputlayer_2_1channel(module, chWeights=[1., 1., 1.] ):
    #Combine the layers using the followng weight:Y = wr R + wg G + wb B 
    #fx. chWeights=[0.299, 0.587, 0.114] )
    #or  chWeights=[1., 1., 1.] )
    # etc
        
    n1,l_rgb = list(module.named_children())[0]
    rgb2gray = 3.0*np.asarray(chWeights) / sum(chWeights)
          
    #create a 1 channel layer to combine the three rgb channels
    conv_2d_gray = nn.Conv2d(1, out_channels=l_rgb.out_channels, kernel_size=l_rgb.kernel_size, 
                             stride=l_rgb.stride, padding=l_rgb.padding, bias=l_rgb.bias)
    #print(conv_2d_gray)
    rgb_weight  = l_rgb.weight.data.cpu().numpy()
    gray_weight = conv_2d_gray.weight.data.cpu().numpy()
    strength    = np.zeros(rgb_weight.shape[0])
    for i in range( 0, rgb_weight.shape[0] ):
        gray_weight[i] =  (chWeights[0]*rgb_weight[i,0] + chWeights[1]*rgb_weight[i,1] + chWeights[2]*rgb_weight[i,2] )
#        strength[i]    = np.sum(np.abs(gray_weight[i]))

    #sort the filter so the stroomng filter ar placede first (for visualization purposes)
#    ix_sort = np.argsort(-strength)
#    conv_2d_gray.weight = torch.nn.Parameter( torch.from_numpy(gray_weight[ix_sort]) )   
    conv_2d_gray.weight = torch.nn.Parameter( torch.from_numpy(gray_weight) )

    #freeze the gray layer
    set_trainable(conv_2d_gray,False)            
            
    #extract all but the first layer
    m_children = list(module.children())[1:]
            
    #insert a the new gray first
    m_children.insert(0,conv_2d_gray)
    
    #return the module that takes a grayscale image as input
    module = nn.Sequential( *m_children )
    print(module)
    return module

def inputlayer_3_2_3_channels(module, chWeights=[1., 1., 1.] ):
    n1,l_rgb = list(module.named_children())[0]
    rgb2gray = 3.0*np.asarray(chWeights) / sum(chWeights)
          
    rgb_weight  = l_rgb.weight.data.cpu().numpy()
    for i in range( 0, rgb_weight.shape[0] ):
        rgb_weight[i,0] = chWeights[0]*rgb_weight[i,0] 
        rgb_weight[i,1] = chWeights[1]*rgb_weight[i,1]
        rgb_weight[i,2] = chWeights[2]*rgb_weight[i,2] 
        
    return module

#Small classs to intercept the instantiation of the model in cnn_create and convert the input filter to grayscale 
class ModelAdapter:
    def __init__( self, arch ):self.arch = arch
        
    def __call__(self, pretrained): 
        module = self.arch(pretrained)
        module = self.customizeNet(module,pretrained)
        return module   
    
    @abstractmethod
    def customizeNet(self, module, pretrained): 
        return module
