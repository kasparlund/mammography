import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 

# Example
# ixCutStart=5
# ixCutEnd=15
# dfMeanStd = smoothHistogram( bins, mHist, sdHist, ixCutStart=ixCutStart, ixCutEnd=ixCutEnd )
# dfMeanStd.insert(1,"histMean",mHist)
# dfMeanStd.insert(2,"histSD",sdHist)
# display(dfMeanStd.head())
# display(dfMeanStd.tail())
# plotSmoothHistogram( dfMeanStd )

def smoothHistogram( bins, hist, sd, ixCutStart, ixCutEnd ):
    #remove extrem values at the end and start of the histogram
    h_clean  = hist.copy()
    sd_clean = sd.copy()
    
    h_clean[0:ixCutStart]  = np.linspace(0,hist[ixCutStart],ixCutStart)
    h_clean[-ixCutEnd-1:]  = np.linspace(hist[-ixCutEnd],0, ixCutEnd+1)
    sd_clean[0:ixCutStart] = np.linspace(0,sd_clean[ixCutStart],ixCutStart)
    sd_clean[-ixCutEnd-1:] = np.linspace(sd_clean[-ixCutEnd],0, ixCutEnd+1)

    #fit the histogram and force the start and end counts to zero
    p  = np.polyfit(bins, h_clean, deg=13)
    hs = np.polyval(p, bins)
    hs[0] = hs[-1] = 0
       
    data = pd.DataFrame(columns=["bins", "hist_smoothed", "hist_no_outliers", "histSD_no_outliers"] )
    data.bins=bins
    data.hist_smoothed = hs
    data.hist_no_outliers = h_clean
    data.histSD_no_outliers = sd_clean
    return data

def plotSmoothHistogram( df  ):
    fig = plt.figure(figsize = (12,5)) 
    ax1 = fig.add_subplot(111)
    ax1.set_title("histograms mean with shading of 1*sd, 2*sd")

    #ax1.fill_between(df.bins, df.hist_smoothed+df.histSD_no_outliers, df.hist_smoothed-df.histSD_no_outliers,facecolor='silver', alpha=0.25)
    #ax1.fill_between(df.bins, df.hist_smoothed+2*df.histSD_no_outliers, df.hist_smoothed-2*df.histSD_no_outliers, facecolor='silver', alpha=0.25)
    ax1.plot( df.bins, df.hist_no_outliers, color="b")
    #ax1.scatter( bins, h_clean, s=1, color="g")
    ax1.plot( df.bins, df.hist_smoothed, linewidth=1, color="r")
