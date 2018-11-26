from fastai import *
from fastai.vision import *

import numpy as np
import matplotlib.pyplot as plt 
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import colorcet as cc
     
def getColorBoundsNorm(cmap, codes ):
    cmap     = mpl.cm.get_cmap(cmap) if type(cmap)==str else cmap
    cmaplist = [cmap(i) for i in range(len(codes))]
    cmap     = LinearSegmentedColormap.from_list('cmap for codes', cmaplist, len(codes))
    bounds   = np.arange(len(codes)+1)
    norm     = mpl.colors.BoundaryNorm(bounds, cmap.N)
    return cmap, bounds, norm

def plotImageMaskMosaic( ims, masks, codes, cmap="tab20", figsize=(9,12) ):
    nb    = len(ims)
    ncols = int(np.sqrt(nb))
    nrows = int(math.ceil(nb/ncols))

    fig  = plt.figure(figsize=figsize)
    
    cmap, bounds, norm = getColorBoundsNorm( cmap, codes )

    gs = gridspec.GridSpec(nrows, ncols, height_ratios=np.zeros(nrows)+ 2, wspace=0.0, hspace=0.0)
    
    for i in range(nb):
        inner = gridspec.GridSpecFromSubplotSpec(2, 1,subplot_spec=gs[i], wspace=0.0, hspace=0.0)
        
        ax = plt.subplot(inner[0])
        ax.axis('off')
        ax.imshow(ims[i])

        ax = plt.subplot(inner[1])
        ax.axis('off')
        ax.imshow(masks[i],norm=norm, cmap=cmap)

def showColorEncoding(codes, cmap="tab20", ax=None) :
    if ax is None: fig, ax = plt.subplots(figsize=(1, 5))

    cmap, bounds, norm = getColorBoundsNorm( cmap, codes )
    cb3 = mpl.colorbar.ColorbarBase(ax,cmap=cmap,norm=norm, boundaries=[-10] + bounds + [10],
                                    ticks=bounds, spacing='uniform', orientation='vertical')
    cb3.set_label('color encoding of class codes')
    cb3.ax.yaxis.set_label_coords(1.25, .5)
    cb3.set_ticklabels(codes)
    cb3.set_ticks(0.5+bounds)
    cb3.ax.yaxis.set_ticks_position('left')

def showColorbar(cmap=cc.cm.linear_grey_0_100_c0, title="grayscale", ax=None) :
    if ax is None: fig, ax = plt.subplots(figsize=(1, 5))

    cb3 = mpl.colorbar.ColorbarBase(ax,cmap=cmap,orientation='vertical')
    cb3.set_label(title)
    cb3.ax.yaxis.set_label_coords(1.1, .5)
    cb3.ax.yaxis.set_ticks_position('left')

def plot_xy_preds( x, y, y_p, preds, codes, cmap, cmap_preds, figWidth=16 ):
    npreds     = preds[0].shape[0]
    widthRatio = float(npreds)/3
    
    preds      = np.nan_to_num(preds)
    vmin       = np.min(preds.flatten())-1e-6
    vmax       = np.max(preds.flatten())+1e-6
    preds      = np.clip(preds,0,vmax)
    preds_norm = mpl.colors.Normalize(vmin=0.,vmax=vmax )
    
    fig  = plt.figure(figsize=(figWidth, len(x)*figWidth/(3.+npreds)  ) )
    
    cmap, bounds, norm = getColorBoundsNorm( cmap, codes )

    nrows      = len(x)
    gs = gridspec.GridSpec(nrows, 2, width_ratios=[1,widthRatio], wspace=0.05, hspace=0.01)
    
    for i in range(nrows):
        g_left  = gridspec.GridSpecFromSubplotSpec(1, 3,      subplot_spec=gs[i,0], wspace=0.01, hspace=0.0)
        g_right = gridspec.GridSpecFromSubplotSpec(1, npreds, subplot_spec=gs[i,1], wspace=0.01, hspace=0.0)
        
        ax = plt.subplot(g_left[0])
        ax.axis('off')
        ax.imshow(x[i])

        ax = plt.subplot(g_left[1])
        ax.axis('off')
        ax.imshow(y[i],norm=norm, cmap=cmap)
        
        ax = plt.subplot(g_left[2])
        ax.axis('off')
        ax.imshow(y_p[i],norm=norm, cmap=cmap)
        
        for j in range(npreds):
            ax = plt.subplot(g_right[j])
            ax.axis('off')
            ax.imshow(preds[i][j],norm=preds_norm, cmap=cmap_preds) 

def getIO(learn, n, ds_type:DatasetType=DatasetType.Valid, class_mask=None):
    "returns n rows of x, y, predicted y, prediction pr class (classes,width,height)"
    xs,ys,ps,pcs=[],[],[],[]
    ds = learn.dl(ds_type).dataset
    for i in range(min(n,len(ds))):
        x,y = ds[i]
        xs.append(image2np(x.px))
        ys.append(image2np(y.px))
        
        y_p, y_p2, p_prClass = learn.predict(x)
        
        ps.append( image2np(y_p.px) )
        p_prClass = p_prClass.numpy()
        pcs.append( p_prClass )
        
    return xs,ys,ps,pcs          

def plotPreds(learn, nrows, codes, code_weights, cmap=plt.cm.tab10, cmap_gray=cc.cm.linear_grey_0_100_c0, ds_type=DatasetType.Valid ):
    class_mask = torch.from_numpy( (code_weights > 0).astype(np.float32) )
    xs,ys,ps,pcs = getIO(learn, nrows, ds_type=ds_type, class_mask=class_mask)
    fig,ax = plt.subplots(nrows=1,ncols=2, figsize=(14,4))
    showColorEncoding(codes, cmap=cmap, ax=ax[0])    
    showColorbar(cmap=cmap_gray, title="grayscale", ax=ax[1])
    plot_xy_preds(xs, ys, ps, np.asarray(pcs), codes, cmap=cmap, cmap_preds=cmap_gray, figWidth=16)
    
"""
def plotTensor( im ):       
    colorMap = plt.cm.gray if im.ndim == 2 else None 
    
    fig = plt.figure(figsize=(8,8)) 
    plt.imshow(im, cmap=colorMap)    
    plt.show()
    
def createFilterImage( weights ) :   
    #scale to 0-1 and put the channel in the last as expect by matplotlib
    weights = 0.5*np.copy(weights+1.0)
    weights = np.rollaxis(weights, 1, 4)
    
    filter_dim = weights.shape[1]
    dim = int( 0.5+ np.sqrt( weights.shape[0] ) ) * filter_dim
    
    im = np.zeros([dim,dim,weights.shape[3]])
    for i in range(8):
        io = i*filter_dim
        for j in range(8):
            jo = j*filter_dim            
            i_filter = i*(filter_dim+1) + j
            #print( "\ni_filter: {}:  i_off: {} - {},  j_off: {} - {}",  i_filter, io, (io+filter_dim), jo,(jo+filter_dim) )
            im[io:(io+filter_dim), jo:(jo+filter_dim), :] = weights[i_filter,:,:,:]

    im = np.squeeze(im)        
    return im, filter_dim

def getLayers( model ): 
    layers=[]
    for g in list( model.children() ):
        layers.extend( list(g.children()) )
    return layers

def WeightsAsImage(model) :
    layers = getLayers(model)
    l      = layers[0]
    
    w = l.weight.data.cpu().numpy()
    im, filter_dim = createFilterImage(w)
    return im

   
def printWeight( model, i0=0, i1=1 ):
    layers = getLayers(model)
 
    for l in layers[i0:i1]:        
        print("meatadata for layer: ", l)
        w = l.weight.data.cpu().numpy()
        im, filter_dim = createFilterImage(w)
        print("One image with all n filterweight with size ", w.shape[0], filter_dim, filter_dim )
        #print("\nweights.shape :", w.shape, " - weights:\n", w)
        plotTensor(im)
    
"""
        