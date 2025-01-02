# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 17:56:48 2023

Code to create synthetic data representing gridded ratio scale measurements and estimate
agreement between them using proposed extensions of measures typically used fot categorical measurements, i.e.:
    - cont. Jaccard
    - cont. Precision
    - cont Recall
    - cont. F1 score
and other error and association measures typically used for continuous measurement (MAE, MAPE, rho...).

@author: Katarzyna Krasnodębska, Institute of Geography and Spatial Organization Polish Academy of Sciences

"""

#%% Import dempendencies
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import patches
from scipy.ndimage import gaussian_filter #, zoom
import os
from matplotlib.patches import Patch
import matplotlib as mpl
from PIL import Image
from libpysal.weights import lat2W
from esda.moran import Moran, Moran_Local

#%% Set plotting parameters
mpl.rcParams['hatch.linewidth'] = 0.1
plt.rcParams["font.family"] = "Arial"
font = {'family' : 'Arial',
        'size'   : 6}
matplotlib.rc('font', **font)
cm = 1/2.54  # centimeters in inches
_cmap = 'turbo'

#%% Assign root directory
root =  r'.\results'
if not os.path.exists(root): os.makedirs(root)

#%% Functions to measure agreement

def RMSD(pred,ref) -> float:
    """ Root Mean Square Deviation """
    if pred.shape != ref.shape:
        raise Exception("Arrays must have the same shape. Found {} and {}".format(pred.shape, ref.shape))
    return np.sqrt(np.mean((pred-ref)**2))

def MAD(pred,ref) -> float: # from Pontius 2022 Metrics that make a difference
    """ Mean Absolute Deviation """
    if pred.shape != ref.shape:
        raise Exception("Arrays must have the same shape. Found {} and {}".format(pred.shape, ref.shape))
    return np.mean(np.absolute(np.subtract(pred,ref)))

def MD(pred,ref) -> float: # from Pontius 2022 Metrics that make a difference
    """ Mean Absolute Deviation """
    if pred.shape != ref.shape:
        raise Exception("Arrays must have the same shape. Found {} and {}".format(pred.shape, ref.shape))
    return np.subtract(np.mean(pred),np.mean(ref))

def rho(pred,ref) -> float:
    """ Pearson's correlation coefficient """
    if pred.shape != ref.shape:
        raise Exception("Arrays must have the same shape. Found {} and {}".format(pred.shape, ref.shape))

    rho = np.corrcoef(pred.astype(np.float32), ref.astype(np.float32))[0,1]
    return rho

def contJaccard(pred,ref) -> float:
    """ Continuous Jaccard """
    if pred.shape != ref.shape:
        raise Exception("Arrays must have the same shape. Found {} and {}".format(pred.shape, ref.shape))
    return np.sum(np.minimum(pred, ref))/np.sum(np.maximum(pred, ref))

def contRecall(pred,ref) -> float:
    """ Continuous Recall """
    if pred.shape != ref.shape:
        raise Exception("Arrays must have the same shape. Found {} and {}".format(pred.shape, ref.shape))
    return np.sum(np.minimum(pred, ref))/np.sum(ref[:])

def contPrecision(pred,ref) -> float:
    """ Continuous Precision """
    if pred.shape != ref.shape:
        raise Exception("Arrays must have the same shape. Found {} and {}".format(pred.shape, ref.shape))
    return np.sum(np.minimum(pred, ref))/np.sum(pred[:])

def fscore(_precision, _recall, beta):
    """ F-Score measure for selected beta """
    if all([_precision==0, _recall==0]):
        return np.nan
    else:
        return (1+beta*beta)*_precision*_recall/((beta*beta*_precision)+_recall)
    
#%% Functions to generate synthetic data
def gaussian_kernel(l, sig, mu):
    """\
    creates gaussian Gaussian window with side length `l` and a sigma of `sig`
    """
    n = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square((n-mu)/sig))
    kernel = np.outer(gauss, gauss)
    return kernel#/np.sum(kernel) #- if divided, it returns normalized Gaussian kernel

def polycentric_landscape(root_landscape, _l, _sig, _mu, _base_value, adjust=False):
    land_interim = root_landscape
    _sig=100

    blob_no=4
    blobs_s = np.array([0.95, 0.85, 0.85, 0.8])  
    blobs_x = [ int(x) for x in np.array([0.325, -0.3, -0.35, 0.4])*_l]
    blobs_y = [ int(x) for x in np.array([-0.325, -0.3, 0.35, 0.4])*_l]
    
    for blob_id in range(blob_no):
        a_blob = gaussian_kernel(_l, _sig*blobs_s[blob_id], _mu) + _base_value
        a_blob = a_blob *blobs_s[blob_id]*0.75
        a_blob = np.roll(np.roll(a_blob, blobs_y[blob_id], axis=0), blobs_x[blob_id], axis=1)
        land_interim = sum([land_interim, a_blob])
    
    if adjust:
        # Adjust the data volume
        adj = np.sum(root_landscape) / np.sum(land_interim)
        land_interim = land_interim * adj
    
    return land_interim
    
def landscape_composite(land1, land2, land1_w):  #assign weight to land 1: 0-1    
    # Stack two landscape
    land_s = sum([land1*land1_w, land2*(1-land1_w)])
    if upper_bounded == True:  land_s[land_s>1] = 1 
    # Adjust the data volume
    # adj = np.sum(land1) / np.sum(land2a)
    # land2a = land2a * adj
    # print(land2a.max())
    
    # Remove some lower values to create zeros 
    land_trim = land_s - land_s.max()*0.3
    # Add lower bound
    land_composite = np.where(land_trim>0,land_trim,0)
    # Rescale to 0-1
    land_composite = land_composite/land_composite.max()
    # Check the output
    plotLandscape('composite landscape', land_composite)
    print('composite landscape min:', land_composite.min())
    print('composite landscape max:', land_composite.max())
    return land_composite

def fake_surface(low, high, sigma, ref_land, adjust=False):
    '''Super basic random terrain function.
    
    This function generates a uniform random surface,
    and applies a sequence (x and y) of one-dimensional
    convolution filters; effectively increasing the spatial
    autocorrelation of the surface relative to sigma.
    
    Paramters
    ---------
    dim : tuple
        The x and y dimensions of the 'terrain' grid.
    low : numeric
        The lowest possible simulated elevation value.
    high : numeric
        The highest possible simulated elevation value.
    sigma : numeric
        The variance of the gaussian kernel. Controls
        the 'smoothness' of the simulated surface.
        Values between 1 and 3 are probably good for
        'small' surfaces.
        
    Returns
    -------
    out : ndarray
        A spatially autocorrelated random 'terrain' surface
        of a given dimension.
    '''
    dim = ref_land.shape
    r = rng1.uniform(low, high, size=np.multiply(*dim)).reshape(dim)
    v = gaussian_filter(r, sigma=sigma, truncate=10)
    # Generate blurr
    filter_blurred_v = gaussian_filter(v, sigma/2)
    v = v - filter_blurred_v
    # Normalize array to min and max values
    v = (v - v.min()) / (v.max() - v.min())
    if adjust:
        # Adjust
        v = v * np.sum(ref_land) / np.sum(v)
    
    return v

#%% Functions to generate experiment results

def experiment(landscape, abs_bias, rel_bias, noise_sds, noise_arr):
    land_1D = landscape.flatten()
    obs = np.copy(land_1D )
    pred = np.copy(land_1D )
    # Add systematic additive bias to each prediction 
    pred_abs_bias = np.repeat([pred], len(abs_bias), axis=0) + abs_bias[:, None]
    # Change negative values to zero
    pred_abs_bias[pred_abs_bias<0]=0
    if upper_bounded is True:
        pred_abs_bias[pred_abs_bias>1]=1
        print('upper_bounded: %s'%upper_bounded)
    print('max value of pred_abs_bias: %s'%pred_abs_bias.max())
    # Calculate agreement measures for each bias
    cont_jaccard_abs = [ contJaccard(pred_bias, obs) for pred_bias in pred_abs_bias]
    cont_precision_abs = [ contPrecision(pred_bias, obs) for pred_bias in pred_abs_bias]
    cont_recall_abs = [ contRecall(pred_bias, obs) for pred_bias in pred_abs_bias]
    RMSD_abs = [RMSD(pred_bias, obs) for pred_bias in pred_abs_bias]
    MAD_abs = [MAD(pred_bias, obs) for pred_bias in pred_abs_bias]
    MD_abs = [MD(pred_bias, obs) for pred_bias in pred_abs_bias]
    rho_abs = [rho(pred_bias, obs) for pred_bias in pred_abs_bias]
    fscore_abs = [fscore(contPrecision(pred_bias, obs),contRecall(pred_bias, obs), beta=1) for pred_bias in pred_abs_bias]
    
    # Add systematic multiplicative bias to each prediction 
    pred_rel_bias = np.repeat([pred], len(rel_bias), axis=0) * rel_bias[:, None]
    # Calculate agreement measures for each bias
    cont_jaccard_rel = [ contJaccard(pred_bias, obs) for pred_bias in pred_rel_bias]
    cont_precision_rel = [ contPrecision(pred_bias, obs) for pred_bias in pred_rel_bias]
    cont_recall_rel = [ contRecall(pred_bias, obs) for pred_bias in pred_rel_bias]
    RMSD_rel = [RMSD(pred_bias, obs) for pred_bias in pred_rel_bias]
    MAD_rel = [MAD(pred_bias, obs) for pred_bias in pred_rel_bias]
    MD_rel = [MD(pred_bias, obs) for pred_bias in pred_rel_bias]
    rho_rel = [rho(pred_bias, obs) for pred_bias in pred_rel_bias]
    fscore_rel = [fscore(contPrecision(pred_bias, obs),contRecall(pred_bias, obs), beta=1) for pred_bias in pred_rel_bias]
    
    # Add noise to each prediction, with mean = 0 and increasing std
    pred_noise = np.repeat([pred], len(noise_arr), axis=0) + noise_arr#[:, None]
    # Cut values lower than zero and greater than 1
    pred_noise[pred_noise<0]=0
    if upper_bounded is True:
        pred_noise[pred_noise>1]=1
    # Calculate agreement measures for each bias
    cont_jaccard_noise = [ contJaccard(pred_n, obs) for pred_n in pred_noise]
    cont_precision_noise = [ contPrecision(pred_n, obs) for pred_n in pred_noise]
    cont_recall_noise = [ contRecall(pred_n, obs) for pred_n in pred_noise]
    RMSD_noise = [RMSD(pred_bias, obs) for pred_bias in pred_noise]
    MAD_noise = [MAD(pred_bias, obs) for pred_bias in pred_noise]
    MD_noise = [MAD(pred_bias, obs) for pred_bias in pred_noise]
    rho_noise = [rho(pred_bias, obs) for pred_bias in pred_noise]
    fscore_noise = [fscore(contPrecision(pred_bias, obs),contRecall(pred_bias, obs), beta=1) for pred_bias in pred_noise]
    
    return [
        [cont_jaccard_abs,cont_precision_abs, cont_recall_abs, fscore_abs, RMSD_abs, MAD_abs, MD_abs, rho_abs, abs_bias],
        [cont_jaccard_rel,cont_precision_rel, cont_recall_rel, fscore_rel, RMSD_rel, MAD_rel, MD_rel, rho_rel, rel_bias],
        [cont_jaccard_noise,cont_precision_noise, cont_recall_noise, fscore_noise, RMSD_noise, MAD_noise, MD_noise, rho_noise, noise_sds]
        ]

#%% Functions to plot the results
def plotLandscape(landscape_name, land1, _vmin=0, _vmax=1, thr=None, box=True):
    f, ax = plt.subplots(dpi=300, figsize=(6.5*cm, 5*cm) )
    im=plt.imshow(land1, interpolation='none', vmin=_vmin, vmax=_vmax, cmap = _cmap) 
    if box is True:
        if thr is not None:
            atext = 'sum: %s\ncutoff:  %s'%(int(land1.sum()), thr)
        else:
            atext = 'sum: %s'%(int(land1.sum()))
        t = plt.text(.08,.9,atext,
                          fontsize=8, 
                          ha='left',va='center',transform = ax.transAxes)
        t.set_bbox(dict(facecolor='white', alpha=0.75, edgecolor='white'))
    plt.title(landscape_name, fontsize=8)
    plt.colorbar(im)
    plt.show()

def plot_measures_1landscape(the_landscape):
    fig_h = 2.5
    l_cols = 2
    _lw=0.25
    
    _ymin, _ymax=-.6,1.05
    nticks1=6
    nticks2=9
    
    r_list = experiments[the_landscape]
    # First plot the landscape on a seperate row
    fig, axs = plt.subplots(nrows=1,ncols=1,figsize = (3,fig_h), dpi=600, sharey=True, sharex=True)
    aland = landscapes[the_landscape]
    axs.imshow(aland, interpolation='none', vmin=0, vmax=1, cmap = _cmap)
    axs.set_ylabel(the_landscape, size=8) #.replace(" ","\n")
    axs.set_xticks([])
    axs.set_yticks([])
    axs.set_title('observed', size=8)
    plt.show()
    
    # Second - charts with NEW measures
    fig, axs = plt.subplots(nrows=1,ncols=3,figsize = (fig_h*1.4,fig_h*0.6), dpi=600, 
                            sharey=True)#, sharex=True) 
    #cont_jaccard,cont_precision, cont_recal, fscore, RMSD, MAD, MD, rho, bias 
    for ie, exp in enumerate(r_list):
        # Plot Jaccard
        axs[ie].plot(exp[0], '-yo', 
                ms=4, lw=0.2, label = 'cont. Jaccard')
        # Plot Precision
        axs[ie].plot(exp[1], '-go', 
                ms=0.8, lw=0.2, label = 'cont. Precision')
        # Plot Recall
        axs[ie].plot(exp[2], '-ro', 
                ms=0.8, lw=0.2, label = 'cont. Recall')
        # Plot F1 score
        axs[ie].plot(exp[3], '--ks', 
                lw=0.2, 
                ms=2, markerfacecolor="black",
                markeredgecolor='black', markeredgewidth=0.2,
                label = 'cont F1-score')            
        axs[ie].set_xticklabels([])    
    
    for ie, exp in enumerate(r_list): 
        ticks = exp[-1]
        ticks_2 = [ np.round(ticks[x],2) if x%5==0 else '' for x in range(len(ticks)) ]
        axs[ie].set_xticks(np.arange(len(ticks_2)), 
                      ticks_2, fontsize=8, rotation='horizontal')
        
    titles = ['additive bias','relative bias','random noise'] 
    cols = ['(a)','(b)','(c)','(d)','(e)','(f)']
    for a, ax in enumerate(axs):   
        ax.set_title(cols[a]+' '+titles[a], size=8)

    for a, ax in enumerate(axs): 
        ax.set_ylim([0, _ymax])
        # ax.set_ylabel(list(landscapes.keys())[a], size=8)
        ax.set_yticks(np.linspace(0,1,nticks1), np.round(np.linspace(0,1,nticks1),2), fontsize=8)
        
    # Add type of bias to columns
    bias_names = ['bias','bias',r'$\sigma$']
    for ib, b in enumerate(bias_names): axs[ib].set_xlabel(b, fontsize=8)

    # Legend goes above the plot
    axs[1].legend(loc='upper center', fontsize=8,edgecolor='None',
             bbox_to_anchor=(0.5, 1.6),fancybox=False, shadow=False, ncol=l_cols)

    plt.show()
    
    #####
    # Plot with OLD measures
    fig, axs = plt.subplots(nrows=1,ncols=3,figsize = (fig_h*1.4,fig_h*0.6), dpi=600, sharey=True)#, sharex=True) 
    #cont_jaccard,cont_precision, cont_recal, fscore, RMSD, MAD, MD, rho, bias
    _ms=3.5
  
    for ie, exp in enumerate(r_list):
        # Plot RMSD
        axs[ie].plot(np.array(exp[4]), '--s',
                lw=_lw, 
                ms=_ms, markerfacecolor="None",
                markeredgecolor='blue', markeredgewidth=0.2,
                label = 'RMSE')
        # Plot MAD
        axs[ie].plot(np.array(exp[5]), '--gs',
                lw=_lw, 
                ms=2, markerfacecolor="g",
                markeredgecolor='g', markeredgewidth=0.2,
                label = 'MAE')
        # Plot MD
        axs[ie].plot(np.array(exp[6]), '--r^',
                lw=_lw, 
                ms=_ms, markerfacecolor="None",
                markeredgecolor='red', markeredgewidth=0.2,
                label = 'ME')
        print('MD', np.array(exp[6]))
        # Plot RHO
        axs[ie].plot(exp[7], '--ko', 
                lw=_lw, 
                ms=_ms, markerfacecolor="None",
                markeredgecolor='black', markeredgewidth=0.2,
                label = "r")
        
        axs[ie].set_xticklabels([])    
    
    for ie, exp in enumerate(r_list): 
        ticks = exp[-1]
        ticks_2 = [ np.round(ticks[x],2) if x%5==0 else '' for x in range(len(ticks)) ]
        axs[ie].set_xticks(np.arange(len(ticks_2)), 
                      ticks_2, fontsize=8, rotation='horizontal')
        
    titles = ['additive bias','relative bias','random noise'] 
    for a, ax in enumerate(axs):   
        ax.set_title(cols[a+3]+' '+titles[a], size=8)

    for a, ax in enumerate(axs): 
        ax.set_ylim([_ymin, _ymax])
        ax.set_yticks(np.linspace(_ymin,1,nticks2), np.round(np.linspace(_ymin,1,nticks2),2), fontsize=8)
        
    # Add type of bias to columns
    for ib, b in enumerate(bias_names): axs[ib].set_xlabel(b, fontsize=8)

    # Legend goes above the plot
    axs[1].legend(loc='upper center', fontsize=8,edgecolor='None',
             bbox_to_anchor=(0.5, 1.6),fancybox=False, shadow=False, ncol=3)
    # plt.setp(ax.spines.values(), lw=0.2, color='k', alpha=1)
    plt.savefig(os.path.join(root, 'agreement-measures-1landscape.png'))
    plt.show()
    
def add_double_arrow(afig, ax1, ax2, scale_f):
    # Add arrow    
    arrow_pad = 180
    line_pad=15
    lw=1
    
    top_line = patches.ConnectionPatch(
    [l_vhr-arrow_pad/2,l_vhr/2+line_pad],
    [arrow_pad/2,l_vhr/2+line_pad],
    coordsA=ax1.transData,
    coordsB=ax2.transData,
    arrowstyle="-",  
    linewidth=lw,
    edgecolor='black',
    )
    bottom_line = patches.ConnectionPatch(
    [l_vhr-arrow_pad/2,l_vhr/2-line_pad],
    [arrow_pad/2,l_vhr/2-line_pad],
    coordsA=ax1.transData,
    coordsB=ax2.transData,
    arrowstyle="-", 
    linewidth=lw,
    edgecolor='black',
    )
    double_arrow = patches.ConnectionPatch(
    [l_vhr-arrow_pad,l_vhr/2],
    [arrow_pad,l_vhr/2],
    coordsA=ax1.transData,
    coordsB=ax2.transData,
    arrowstyle="<|-|>",  # "normal" two-side arrow
    mutation_scale=scale_f,  # controls arrow head size
    linewidth=lw,
    facecolor='white',
    edgecolor='k',
    )
    middle_line = patches.ConnectionPatch(
    [l_vhr-arrow_pad/2,l_vhr/2],
    [arrow_pad/2,l_vhr/2],
    coordsA=ax1.transData,
    coordsB=ax2.transData,
    arrowstyle="-",  
    linewidth=1.5,
    edgecolor='white',
    )
    
    afig.patches.append(top_line)
    afig.patches.append(bottom_line)
    afig.patches.append(double_arrow)
    afig.patches.append(middle_line)
    
def plot_observed_modelled_maps_1landscape(the_landscape, _vmax):
    
    # Figure with 1 row - for one landscapes - and 6 columns:
    # Figure with 1st, 3rd, 5th column: observation
    # Figure with 2nd, 4th, 6th column: prediction
    fig, axs = plt.subplots(nrows=1,ncols=6,
                            figsize = (20*cm,4*cm), dpi=600, sharey=True, sharex=True)
    
    aland = landscapes[the_landscape]
    # First, third, fifth column shows the 'observed' landscape
    for acol in [0,2,4]:
        im = axs[acol].imshow(aland, interpolation='none', vmin=0, vmax=_vmax, cmap = _cmap)
    
    # Add additive bias to the landscape
    land_1D = aland.flatten()
    pred = np.copy(land_1D)
    # Add systematic additive bias to each prediction 
    pred_abs_bias = np.repeat([pred], len(abs_bias), axis=0) + abs_bias[:, None]
    # Change negative values to zero
    pred_abs_bias[pred_abs_bias<0]=0
    if upper_bounded is True:
        pred_abs_bias[pred_abs_bias>1]=1
    
    # Add multiplicative bias to the landscape
    pred_rel_bias = np.repeat([pred], len(rel_bias), axis=0) * rel_bias[:, None]
        
    # Add noise to the landscape
    cont_noise = aland+noise_vhr[-1].reshape(aland.shape)
    # Cut values lower than zero and greater than 1
    cont_noise[cont_noise<0]=0
    if upper_bounded is True:
        cont_noise[cont_noise>1]=1
    
    _vmax_real = np.round(max(pred_abs_bias[-1].max(), cont_noise.max(), pred_rel_bias[-1].max()),2)
    _vmin=0
    
    # Plot predictions!
    im = axs[1].imshow(pred_abs_bias[-1].reshape(aland.shape), interpolation='none', vmin=_vmin, vmax=_vmax, cmap = _cmap)
    im = axs[3].imshow(pred_rel_bias[-1].reshape(aland.shape), interpolation='none', vmin=_vmin, vmax=_vmax, cmap = _cmap)
    im = axs[5].imshow(cont_noise, interpolation='none', vmin=_vmin, vmax=_vmax, cmap = _cmap)  
    
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    cols_titles = {
        0: 'reference',1:'modelled',
        2: 'reference',3:'modelled',#
        4: 'reference',5:'modelled'}#}
       
    # Add observed - modelled arrows
    for icol in [0,2,4]: add_double_arrow(fig, axs[icol], axs[icol+1], 15)
    
    # Add lines in-between columns
    axs[1].plot([1.1, 1.1], [-0.2, 1.2], lw=.5,color='k',transform=axs[1].transAxes, clip_on=False)
    axs[3].plot([1.1, 1.1], [-0.2, 1.2], lw=.5,color='k',transform=axs[3].transAxes, clip_on=False)
    
    # axs[1].vlines(x=1.5, ymin=0, ymax=1, transform = axs[1].transAxes, lw=2)

    # Add title to every column
    for acol in cols_titles.keys(): axs[acol].set_title(cols_titles[acol], size=8)
    # Add type of bias to every pair of columns. Add spaces to fit under the modelled grid
    bias_names = ['absolute bias=%.2f     '%abs_bias[-1],'relative bias=%.2f     '%rel_bias[-1],r'$\sigma=%.2f$'%noise_sds[-1]]
    for ib, b in enumerate(bias_names):
        axs[ib*2+1].set_xlabel(b, fontsize=8)
    
    # add error labels a, b, c
    for t, label in enumerate(['a)','b)','c)']):
        plt.text(axs[t*2].get_position().x0, axs[0].get_position().y1+0.05,
                 label,fontsize=8,fontweight='bold',
                 transform=fig.transFigure, figure=fig)
           
    plt.savefig(os.path.join(root, 'observed-modelled-1landscape.png'), dpi=600)
    plt.show() 

def reshape_array(inarr,out_shape):	
    arr_reshaped = np.array(Image.fromarray(inarr).resize((out_shape[1],out_shape[0]),resample=Image.Resampling.NEAREST))
    return arr_reshaped
 
def plot_modelled_1landscape(the_landscape, _vmax):
    # Plot modelled landscapes, each row shows different kind of bias
    # Modelled bias equal observed is marked with thick border
    fig, axs = plt.subplots(nrows=3,ncols=len(abs_bias),figsize = (20*cm,7*cm), 
                            dpi=600, sharey=True, sharex=True)#,layout='constrained') 
   
    aland = landscapes[the_landscape]
    
    # Add additive bias to the landscape
    land_1D = aland.flatten()
    pred = np.copy(land_1D)
    # Add systematic additive bias to each prediction 
    pred_abs_bias = np.repeat([pred], len(abs_bias), axis=0) + abs_bias[:, None]
    # Change negative values to zero
    pred_abs_bias[pred_abs_bias<0]=0
    if upper_bounded is True:
        pred_abs_bias[pred_abs_bias>1]=1
    # Add multiplicative bias to the landscape
    pred_rel_bias = np.repeat([pred], len(rel_bias), axis=0) * rel_bias[:, None]  
            
    # Plot modelled landscapes
    for m, modelled in enumerate([pred_abs_bias, pred_rel_bias]):
        for ia, a in enumerate(modelled):
            axs[m, ia].imshow(a.reshape(aland.shape), interpolation='none', vmin=0, vmax=_vmax, cmap = _cmap)   
    # Add noise to the landscape
    for acol, anoise in enumerate(noise_vhr):
        cont_noise = aland+anoise.reshape(aland.shape)
        # Cut values lower than zero and greater than 1
        cont_noise[cont_noise<0]=0
        if upper_bounded is True:
            cont_noise[cont_noise>1]=1
        im = axs[m+1, acol].imshow(cont_noise.reshape(aland.shape), interpolation='none', vmin=0, vmax=_vmax, cmap = _cmap)    
            
    cols_titles = {
        0: ['%.2f'%a for a in abs_bias] , #['reference']+['absolute\nbias=%.2f'%a for a in abs_bias]
        1: ['%.2f'%r for r in rel_bias] , #['reference']+['relative\nbias=%.2f'%r for r in rel_bias] 
        2: ['%.2f'%n for n in noise_sds] }#['reference']+['noise of\n$\sigma$=%.2f'%n for n in noise_sds]
    row_titles = ['additive bias','relative bias', 'noise ($\sigma$)'] 
    for ia, axrow in enumerate(axs):
        axrow[0].set_ylabel(row_titles[ia], size=8)
        for a in axrow:
            a.set_xticks([])
            a.set_yticks([])
   
        for acol in range(len(axrow)):
            axrow[acol].set_title(cols_titles[ia][acol], size=8)
        
    # Add major Y label
    axs[0,0].text(-0.4, 1.7, 'd)', ha='left', va='center',weight='bold',
                  transform=axs[0,0].transAxes, fontsize=8) 
    axs[0,0].text(0, 1.7, 'Modelled datasets after modifying the reference dataset with:', 
                  ha='left', va='center', rotation=0,
                  transform=axs[0,0].transAxes, fontsize=8) 
    
    # add colorbar
    fig.subplots_adjust(right=0.85)  # making some room for cbar
    # getting the lower left (x0,y0) and upper right (x1,y1) corners of the last map in top row:
    [[xt10,yt10],[xt11,yt11]] = axs[0,len(axs[0])-1].get_position().get_points()
    # getting the lower left (x0,y0) and upper right (x1,y1) corners of the last map in bottom row:
    [[x10,y10],[x11,y11]] = axs[-1,len(axs[0])-1].get_position().get_points()
    pad = 0.01; width = 0.008
    cbar_ax = fig.add_axes([x11+pad, y10, width, 0.69])
    axcb2 = fig.colorbar(im, cax=cbar_ax, cmap = _cmap)
    _vmax_real = np.round(max(pred_abs_bias[-1].max(), cont_noise.max(), pred_rel_bias[-1].max()),2)
    _ticks = np.linspace(0, _vmax, int(_vmax_real*5+1))
    _labels = [np.round(x,2) for x in np.linspace(0, _vmax_real, int(_vmax_real*5+1))]
    axcb2 = fig.colorbar(im, cax=cbar_ax, ticks=_ticks)
    axcb2.ax.set_yticklabels(_labels, fontsize=8) 
    plt.savefig(os.path.join(root, 'modelled-datasets-1landscape.png'))
    plt.show()

def landscape_autocorrelation(aland):
    w = lat2W(nrows = aland.shape[0],
              ncols=aland.shape[1], 
              rook=False, # queen contiguity
              id_type="string")   
    lm = Moran_Local(aland,w,transformation='r', permutations=999)
    gm = Moran(aland,w,transformation='r', permutations=999, two_tailed=True )
    lmoran_significance = np.reshape(lm.p_sim, aland.shape)
    lmoran_quadrants = np.reshape(lm.q, aland.shape)
    lq = np.where(lmoran_quadrants==3, 5, lmoran_quadrants)
    lq = np.where(lq==4, 3, lq)
    lq = np.where(lq==5, 4, lq)
    # plotLandscape('Local Morans significance', lmoran_significance, _vmax=0.5, box=False)
    
    fig, ax = plt.subplots(1,2, dpi=600, figsize=(8,3))
    ax[0].imshow(lmoran_significance<0.05, interpolation='none', cmap='binary')
    patch = Patch(color='black', label="Local Morans \np_val < 0.05")
    ax[0].legend(handles=[patch], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
    
    values = [1,2,3,4,]
    labels = {1:'HH',2:'LH',3:'HL',4:'LL'}
    im = ax[1].imshow(lq, interpolation='none', cmap='coolwarm_r')
    colors = [ im.cmap(im.norm(value)) for value in values]
    patches = [ Patch(color=colors[i], label="{l} quadrant".format(l=labels[values[i]]) ) for i in range(len(values)) ]
    ax[1].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
    plt.savefig(os.path.join(root, 'autocorrelation-1landscape.png'), dpi=600)
    plt.show()

    print('value of Moran’s I: %.3f'%gm.I)
    
#%% Run the MAIN experiments

# Select metrics to compute
metrics = [['Jaccard', 'cont. Jaccard'], ['Precision', 'cont. Precision'],['Recall','cont. Recall'],
           ['F1 score','F1 score'],
           ['1-RMSD','1-RMSD'],['1-MAD', '1-MAD'],["MCC",r"$\rho$"]]

# Define a large seed for the generator so that results can be reproduced
rng = np.random.default_rng(986488172785971276760594211193410795)
rng1 = np.random.default_rng(986488172785971276760594211193410797)

# Define the lowest value in our data (gte zero)
base_value=0 
# Define size of the high resolution grid: Gaussian window, standard deviation and mean value
l_vhr, sig_vhr, mu_vhr = (1000, 250, 0)

# Define the absolute, relative bias and noise
start_abs, stop_abs = (-1, 1)
start_rel, stop_rel = (0.5, 1.5)
start_noise, stop_noise = (.0, .2)

# Define the runs for bias/noise generation
num = 11 # How fine is the increment in bias
noise_sds = np.linspace(start_noise,stop_noise, num)
abs_bias = np.linspace(start_abs,stop_abs, num)
rel_bias = np.linspace(start_rel,stop_rel, num)
noise_vhr = np.array([ rng.normal(loc = 0, scale = sd, size= l_vhr*l_vhr) for sd in noise_sds]) # check if Generator seed is set!

# Decide if the landscape values should be upper bounded (cut) at 1. Always False!
upper_bounded=False 
# Create interim landscapes
land1_vhr = np.round(gaussian_kernel(l_vhr, sig_vhr, mu_vhr) + base_value,3)
land2_vhr = polycentric_landscape(land1_vhr, l_vhr, sig_vhr, mu_vhr, base_value, adjust=True)
land3_vhr = fake_surface(0, 1, sig_vhr/25, land1_vhr, adjust=True)
# Create the high resolution composite landscape
land_vhr = landscape_composite(land2_vhr, land3_vhr, land1_w = 0.75)
landscapes = {'composite landscape': land_vhr*2}

# Estimate agreement between composite landscape and the same landscape with a bias added
experiments={}
for l in landscapes.keys():
    experiments[l]= experiment(landscapes[l], abs_bias, rel_bias, noise_sds, noise_vhr)#cont_jaccard,cont_precision, cont_recal, fscore_ RMSD, MAD, MD, rho, bias

############################################################################
# MAIN FIGURES
print('MAIN: Experiment 1. Plot observation and prediction maps')
plot_observed_modelled_maps_1landscape('composite landscape', 3)
plot_modelled_1landscape('composite landscape', 3)
                                       
print('MAIN: Experiment 1. Plot observation and prediction charts')
plot_measures_1landscape('composite landscape')

############################################################################
# APPENDIX 
print("Plot autocorrelation")
arr = reshape_array(land_vhr,[100,100]).astype(dtype='float64')
landscape_autocorrelation(arr) 

        







