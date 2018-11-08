import numpy as np 
import pandas as pd
from enum import Enum, auto
import cv2

class HISTOGRAMTYPE(Enum):
    BINS = auto(),
    EVERYVALUE= auto()
    UNIQUE = auto()

class EQC(Enum) :
    HSV  = auto()
    CHX  = auto()    #all channels are mixed into the equalization
        
class EQ(Enum) :
    NO = auto()
    GLOBAL = auto()
    REVERSE = auto()
    TRANSFER = auto()
    TRANSFER_IN_ONE_STEP = auto()

def RGB2GRAY( im ):
    #im = 0.299*R + 0.587*G + 0.114*B
    return 0.299*im[:,:,0] + 0.587*im[:,:,1] + 0.114*im[:,:,2]

class ImageColors:
    def __init__(self, im, eqc):
        self._im=im
        self._eqc = eqc if len(im.shape)==3 and im.shape[2]==3 else EQC.CHX
        self._hsv=None
        
    @property
    def V(self):
        if self._eqc == EQC.HSV :
            if self._hsv is None:
                self._hsv = cv2.cvtColor(self._im, cv2.COLOR_RGB2HSV_FULL )
            return self._hsv[:,:,2]
        else: 
            return self._im
    @V.setter
    def V(self, value): 
        if self._eqc == EQC.HSV :
        #if self._hsv is not None and self.isRGB_:
            self._hsv[:,:,2] = value[:,:]
            self._im = None
        else:
            self._im = value
    @property
    def im(self):
        #if self.isRGB_ and self._im is None:
        if self._eqc==EQC.HSV and self._im is None:
            self._im  = cv2.cvtColor(self._hsv, cv2.COLOR_HSV2RGB_FULL)
            #self._hsv = None
        return self._im
   
class Histogram:
    def __init__(self, hist, vmin, vmax, middelBins=None ):
        #if len(bins) = len(hist)+1 then we set the bins to the center of each bins and discard the rightmost bin value. 
        #else we assume the bns have been centralized
        #if len(bins) == len(hist)+1 :
        #    bins = 0.5*( bins[0:-1] + bins[1:] )
        self._hist = hist
        self._min = vmin
        self._max = vmax
        self._middelBins = middelBins
        
    @staticmethod
    def fromBinIntervals( histType, signal, vmin, vmax, nBins ):
        if histType == HISTOGRAMTYPE.BINS:
            #normal equalization
            hist,bins = np.histogram( signal, np.linspace(vmin, vmax, nBins+1),  (vmin,vmax) )        
            return Histogram( hist, vmin, vmax )
        elif histType == HISTOGRAMTYPE.EVERYVALUE:
            #equalize every unique value with it count
            middelBins,hist = np.unique(signal.flatten(),return_counts=True)
            return Histogram( hist, vmin, vmax, middelBins )
        elif histType == HISTOGRAMTYPE.UNIQUE:
            #equalize unique value. Unique values only contribute with 1
            middelBins = np.unique(signal.flatten())
            hist = np.zeros_like(middelBins)+1
            return Histogram( hist, vmin, vmax, middelBins )

    @property
    def middelBins(self): 
        if self._middelBins is None:
            deltaBin = (self._max-self._min)/len(self._hist)
            self._middelBins = 0.5*deltaBin + np.linspace(self._min,self._max-deltaBin,len(self._hist))
            self._middelBins = self.align(self._middelBins, self.min, self.max)
        return self._middelBins    

    #def binIntervals(self): return np.linspace(self._min,self._max,len(self._hist)+1)
    
    @property
    def min(self): return self._min

    @property
    def max(self): return self._max

    #@property
    #def binRange(self): return (self._min,self._max)

    #scales and offsets the input to the interval [min max]
    def align(self, s, vmin, vmax): return vmin + (s-s.min())* ( (vmax-vmin) / (s.max()-s.min()) )
    
    #@property
    #def alignBins(self): return  self.align(self.middelBins, self.min, self.max)
    @property
    def alignCDF(self): return  self.align(self.cdf, self.min, self.max)
            
    @property
    def hist(self): return self._hist
    #@hist.setter
    #def hist(self, value): self._hist = value

    @property
    def cdf(self):
        self._cdf    = self.hist.cumsum()
        
        self.cdf_min = self._cdf.min() 
        self.cdf_max = self._cdf.max()
        
        self._cdf  = (self._cdf-self.cdf_min) * ((self.max - self.min) / (self.cdf_max-self.cdf_min))
        self._cdf += self.min
        
        self._cdf = self.align(self._cdf, self.min, self.max)
        return self._cdf
    
    #global equalization of signal
    def equalize (self, signal ):
        shape  = signal.shape 
        sf     = signal.flatten()
        
        ix     = np.logical_and( self.min <= sf , sf <= self.max )
        _,ix_u = np.unique(self.cdf, return_index=True)    
        sf[ix] = np.interp(sf[ix], self.middelBins[ix_u], self.cdf[ix_u])

        #_,ix_u = np.unique(self.alignCDF, return_index=True)    
        #sf[ix] = np.interp(sf[ix], self.alignBins[ix_u], self.alignCDF[ix_u])
        
        #b = self.align(self.middelBins,self.middelBins.min(), self.middelBins.max())
        #sf = np.interp(sf, self.middelBins, self.alignCDF)
        
        signal = sf
        signal.shape = shape    
        return signal
    
    #equalize the signal so that the shape of its histogram is the same as for self 
    def equalize_r (self, signal ):
        shape = signal.shape

        sf     = signal.flatten()
        ix     = np.logical_and( self.min <= sf , sf <= self.max )
        _,ix_u = np.unique(self.cdf, return_index=True)    
        sf[ix] = np.interp(sf[ix], self.cdf[ix_u], self.middelBins[ix_u])
        #_,ix_u = np.unique(self.alignCDF, return_index=True)    
        #sf[ix] = np.interp(sf[ix], self.alignCDF[ix_u], self.alignBins[ix_u])
        signal = sf        
        signal.shape = shape    
        return signal

    #equalize the signal so that the shape of its histogram is the same as target
    def equalize_t (self, signal, target ):
            shape = signal.shape
            
            v,ix_u = np.unique(target.cdf, return_index=True)   
            cdf_modified = np.interp(self.cdf, target.cdf[ix_u], target.middelBins[ix_u])
            #v,ix_u = np.unique(target.alignCDF, return_index=True)   
            #cdf_modified = np.interp(self.alignCDF, target.alignCDF[ix_u], target.alignBins[ix_u])
            
            v,ix_u = np.unique(cdf_modified, return_index=True)    
            cdf_modified = self.align( cdf_modified, self.min, self.max )

            sf     = signal.flatten()
            ix     = np.logical_and( self.min <= sf, sf <= self.max )  
            sf[ix] = np.interp(sf[ix], self.middelBins[ix_u], cdf_modified[ix_u])
#            sf[ix] = np.interp(sf[ix], self.alignBins[ix_u], cdf_modified[ix_u])
            signal = sf
            signal.shape = shape    
        
            return signal
         
class Equalizer:
    def __init__(self, eq_method, eqc, histType, vmin=0, vmax=1, nBins=64, transHist=None ):
        self.eq        = eq_method
        self.eqc       = eqc
        self.histType  = histType
        self._min      = vmin
        self._max      = vmax
        self._nBins    = nBins
        self.transHist = transHist
        
    
    def do(self, im ):
        imc = ImageColors(im, self.eqc)
        if self.eq == EQ.NO:
            hist = Histogram.fromBinIntervals( self.histType, imc.V.flatten(), self._min, self._max, self._nBins )
            
        if self.eq == EQ.GLOBAL:
            #forward
            hist = Histogram.fromBinIntervals( self.histType, imc.V.flatten(), self._min, self._max, self._nBins )
            imc.V = hist.equalize( imc.V )
            
        if self.eq == EQ.REVERSE:
            #forward
            hist  = Histogram.fromBinIntervals( self.histType, imc.V.flatten(), self._min, self._max, self._nBins )
            imc.V = hist.equalize( imc.V )
            
            #reverse
            imc.V = hist.equalize_r(imc.V)

        if self.eq == EQ.TRANSFER:
            #forward
            hist  = Histogram.fromBinIntervals( self.histType, imc.V.flatten(), self._min, self._max, self._nBins )
            imc.V = hist.equalize( imc.V )
            
            #reverse
            imc.V = self.transHist.equalize_r(imc.V)
            
        if self.eq == EQ.TRANSFER_IN_ONE_STEP:
            #forward
            hist = Histogram.fromBinIntervals( self.histType, imc.V.flatten(), self._min, self._max, self._nBins )
            
            #Pass the target histogram so it can be use to modify the cdf for global equalization to 
            # a cdf that takes the image direct to the shape of self.transHist
            imc.V = hist.equalize_t( imc.V, self.transHist )
            
        #calculate the histogran of the new image
        hist = Histogram.fromBinIntervals( self.histType, imc.V.flatten(), 0.0, 1.0, self._nBins )
            
        return imc.im, hist   