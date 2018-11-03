# import matplotlib and numpy
import matplotlib.pyplot as plt 
import matplotlib.image as mpimage
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec

from matplotlib import cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable

import colorcet as cc
import numpy as np 
import pandas as pd
import cv2
import math


def plotImage( ax, title, im, vmin, vmax, cmap):
    #ax.tick_params(axis='both', which='major', labelsize=6)
    #ax.tick_params(axis='both', which='minor', labelsize=6)
    ax.set_xticks([])
    ax.set_yticks([])    
    ax.set_title(title,fontsize=8)
    plt.rcParams.update({'font.size':6})
    ax.imshow(im,aspect=im.shape[1]/im.shape[0], vmin=vmin, vmax=vmax, cmap=cmap)

def plotCurve( ax, title, x, y, ymax, cmap):
    ax.tick_params(axis='both', which='major', labelsize=6)
    ax.tick_params(axis='both', which='minor', labelsize=6)
    ax.set_title(title,fontsize=8)
    plt.rcParams.update({'font.size':6})
    if ymax is not None : ax.set_ylim(0,ymax)
    if False == (x.shape == y.shape) : 
        print(f"error: shape af x,y: {x.shape}, {y.shape}")
    else:     
        ax.plot(x,y)

def plotSortedHistograms(dfhists, hStats, vMin, vMax, cmap, selectSortBy):
    fig = plt.figure(figsize=(18, 9))
    fig.suptitle("Sorted histograms",fontsize=12)
    gs0 = gridspec.GridSpec(1, 2)

    lcols = 3
    lrows = math.ceil( hStats.shape[1]/lcols )
    gs00  = gridspec.GridSpecFromSubplotSpec(lrows, lcols, subplot_spec=gs0[0])
    gs01  = gridspec.GridSpecFromSubplotSpec(    1,     1, subplot_spec=gs0[1])

    #plot all the application of all the sorting parameters to the left
    for i in range(lrows):
        for j in range(lcols):
            if j+i*lcols < hStats.shape[1]:
                c    = hStats.columns[j+i*lcols]
                ix   = hStats[c].argsort().values
                ax00 = fig.add_subplot(gs00[i,j])
                ax00.set_xticks([])
                ax00.set_yticks([])    
                plotImage(ax00, f"sorted by {c[1:]}", dfhists.iloc[ix].values, vMin, vMax, cmap )

    #plot the best sorting paramter to the right
    ax01 = fig.add_subplot(gs01[0,0])
    c    = selectSortBy
    ix   = hStats[c].argsort().values      
    ax01.set_xticks([])
    ax01.set_yticks([])    
    plotImage(ax01, f"sorted by {c[1:]}", dfhists.iloc[ix].values, vMin, vMax, cmap )


def plotHistogramImage( path, reader, filenames, figsize, ymax, cmap, equalizer ):
    hists = []

    ncols    = len(filenames    )
    nrows    = 3
    fig,axes = plt.subplots(nrows=nrows, ncols=ncols, figsize = figsize, dpi=100)

    row=0
    for i in range(len(filenames)):
        ax       = axes.flatten()[row*ncols+i]
        im       = reader.open(list( path.glob(f"**/{filenames[i]}.*") )[0])
        im, hist = equalizer.do(im)
        hists.append(hist)
        ax.set_xticks([])
        ax.set_yticks([])    
        plotImage(ax, f"{filenames[i]}", im, 0, 1, cmap )       

    row=1
    for i in range(len(filenames)):
        ax   = axes.flatten()[row*ncols+i]
        hmax = ymax if ymax is not None else 1.1*hists[i].hist.max()
        plotCurve(ax, f"histogram: {filenames[i]}", hists[i].middelBins, hists[i].hist, hmax, cmap )

    row=2
    for i in range(len(filenames)):
        ax   = axes.flatten()[row*ncols+i]
        hmax = ymax if ymax is not None else 1.1*hists[i].cdf.max()
        plotCurve(ax, f"CDF: {filenames[i]}", hists[i].middelBins, hists[i].cdf, hmax, cmap )

def plotRow(axes, subject, im, hist, hmax, cdfMax, cmap):
    plotImage( axes[0], f"Image: {subject}",     im, 0, im.max(), cmap )
    plotCurve( axes[1], f"Histogram: {subject}", hist.middelBins, hist.hist, ymax=hmax, cmap=cmap )
    plotCurve( axes[2], f"CDF: {subject}",       hist.middelBins, hist.cdf,  ymax=cdfMax, cmap=cmap )
#    plotImage( axes[0], f"{subject} image {hist.binRange}", im, 0, im.max(), cmap )
#    plotCurve( axes[1], f"{subject} hist {hist.binRange}",  hist.middelBins, hist.hist, ymax=hmax, cmap=cmap )
#    plotCurve( axes[2], f"{subject} cdf {hist.binRange}",   hist.middelBins, hist.cdf,  ymax=cdfMax, cmap=cmap )

def plotHistograms(path, reader, dfhists, subject, ix_start, nb, figsize, ymax, eq, bins, cmap, imExt=".jpg" ):
    ncols    = nb
    nrows    = 2
    fig,axes = plt.subplots(nrows=nrows, ncols=ncols, figsize = figsize, dpi=100)
    #fig.suptitle(subject,fontsize=12)

    row=0
    x = np.asarray(  [float(i) for i in dfhists.columns] )
    for i in range(nb):
        filename = dfhists.index[ix_start+i]
        im       = reader.open(list( path.glob(f"**/{filename}.*") )[0] )
        im, hist = Equalizer( EQ.NO, bins ).do(im)
        hmax     = ymax if ymax is not None else dfhists.hist.max()
        plotCurve(axes.flatten()[row+i], f"before equalization:{filename}", hist.middelBins, hist.hist, ymax=hmax, cmap=cmap )

    row=1    
        
    for i in range(nb):
        filename = dfhists.index[ix_start+i] 
        im       = reader.open(list( path.glob(f"**/{filename}.*") )[0])
        im, hist = eq.do(im)
        #ixs      = slice(nbCut,len(hist.bins)-nbCut)
        plotCurve(axes.flatten()[row*ncols+i], f"after equalization:{filename}", hist.middelBins, hist.hist, ymax=hmax, cmap=cmap )
