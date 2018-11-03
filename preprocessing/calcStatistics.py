import numpy as np 
import pandas as pd

#estimate the meand and the standard deviation of the distribution and 
#Pearson's mode skewness : https://en.wikipedia.org/wiki/Skewness
# or Pearson's median skewness : https://en.wikipedia.org/wiki/Skewness
def mean_sd(bins, hist):
    bins     = np.asarray(bins)
    hist     = np.asarray(hist)
    histAcc  = np.cumsum(hist)
    median   = bins[ np.argmin( np.abs(histAcc-0.5*histAcc[-1]) ) ]
    
    mean     = sum(bins * hist)/sum(hist)
    diff     = (bins - mean) 
    #se counts has been normalized
    std      = np.sqrt( np.sum(np.square(diff)*hist) / np.sum(hist) )
    
    mc       = np.argmax(hist)
    mode     = bins[mc]
    
    n        = len(hist)
    w        = 9
    w        = w if w < mc and mc < n-w else min(mc,n-mc)
    modeW    = np.sum( bins[mc-w:mc+w+1]*hist[mc-w:mc+w+1] )/(np.sum( hist[mc-w:mc+w+1]) +1e-12 )
    
   # print(f"m÷{mode:.6f}, m*:{dev1:.6f} b:{bins[mc-1:mc+2]} h:{hist[mc-1:mc+2]}")
    modeSkewness   = (mean-mode)/std
    medianSkewness = (mean-median)/std
    return [mean, median, mode, modeW, std, modeSkewness, medianSkewness]

def calcStatistics(bins, histograms, ixCutStart=0, ixCutEnd=0):
    means, medians, modes, modeWs, sds, modeSkewnesses, medianSkewnesses = [],[],[],[],[],[],[]

    # extrem low and heigh values in the histogram are ignored in the following 
    # in order to get the shape of the main part of the histogram
    # The skewness measure we use is shown here: https://en.wikipedia.org/wiki/Skewness
    # skewness is zero for a normal distribution and negative/positive when the histogram 
    # is heigh to the left/right of the mean 
    for i in range(histograms.shape[0]):
        hist = histograms.iloc[i].values[ixCutStart:] if ixCutEnd==0 else hist[ixCutStart:-ixCutEnd]
        
        mean, median, mode, modeW, sd, modeSkewness, medianSkewness = mean_sd(bins,hist )
        means.append(mean)
        sds.append(sd)
        modeWs.append(modeW)
        modes.append(mode)
        medians.append(median)
        modeSkewnesses.append( modeSkewness )
        medianSkewnesses.append( medianSkewness )

    histShapes = pd.DataFrame(index=histograms.index, columns=["hfilenames", "hMean", "hMedian", "hMode", "hModeW", "hSD", "hModeSkewness", "hMedianSkewness"] )
    histShapes.hfilenames = histograms.index
    histShapes.hMean     = np.round(means,5)
    histShapes.hMedian   = np.round(medians,5)
    histShapes.hMode     = np.round(modes,5)
    histShapes.hModeW    = np.round(modeWs,5)
    histShapes.hSD       = np.round(sds,5)
    histShapes.hModeSkewness = np.round(modeSkewnesses,5)
    histShapes.hMedianSkewness = np.round(medianSkewnesses,5)
    return histShapes