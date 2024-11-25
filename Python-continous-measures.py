# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 17:56:48 2024
continuous-jaccard - Precision, Recall, F-score, and Jaccard Index for continuous, ratio-scale measurements

An approach to extend commonly used agreement measures estimated from a confusion matrix to non-negative ratio-scale attributes.
Useful for comparing agreement of gridded magnitude estimates with bounded, dimensionless measures.

related publication: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4865121
@author: Katarzyna Krasnodębska, Martino Pesaresi
"""

# Import dependencies
import numpy as np
import random
import matplotlib.pyplot as plt

# Define functions
def RMSE(pred,ref) -> float:
    """ Root Mean Square Error """
    if pred.shape != ref.shape:
        raise Exception("Arrays must have the same shape. Found {} and {}".format(pred.shape, ref.shape))
    return np.sqrt(np.mean((pred-ref)**2))

def MAE(pred,ref) -> float: # from Pontius 2022 Metrics that make a difference
    """ Mean Absolute Error """
    if pred.shape != ref.shape:
        raise Exception("Arrays must have the same shape. Found {} and {}".format(pred.shape, ref.shape))
    return np.mean(np.absolute(np.subtract(pred,ref)))

def ME(pred,ref) -> float: # from Pontius 2022 Metrics that make a difference
    """ Mean Error """
    if pred.shape != ref.shape:
        raise Exception("Arrays must have the same shape. Found {} and {}".format(pred.shape, ref.shape))
    return np.subtract(np.mean(pred),np.mean(ref))

def MAPE(pred,ref) -> float:
    """ Mean Absolute Percentage Error """
    if pred.shape != ref.shape:
        raise Exception("Arrays must have the same shape. Found {} and {}".format(pred.shape, ref.shape))
    pred=np.round(pred,4)
    ref = np.round(ref,4)    
    pred_mask = np.where(ref>0, pred, np.nan)
    ref_mask = np.where(ref>0, ref, np.nan)
    mape = np.nanmean(np.absolute(np.divide(np.subtract(pred_mask,ref_mask),ref_mask)))*100 # [%]
    
    return mape#np.mean(np.abs((ref - pred) / ref)) 

def rho(pred,ref) -> float:
    """ Pearson's correlation coefficient """
    if pred.shape != ref.shape:
        raise Exception("Arrays must have the same shape. Found {} and {}".format(pred.shape, ref.shape))

    rho = np.corrcoef(pred.astype(np.float32).flatten(), ref.astype(np.float32).flatten())[0,1]
    
    return rho

def Slope(pred,ref) -> float:
    """ Slope of the regression line """
    if pred.shape != ref.shape:
        raise Exception("Arrays must have the same shape. Found {} and {}".format(pred.shape, ref.shape))

    slope = np.polyfit(pred.astype(np.float32).flatten(), ref.astype(np.float32).flatten(), deg=1)[0]

    return slope

def contJaccard(pred,ref) -> float:
    """ Continuous Jaccard """
    if pred.shape != ref.shape:
        raise Exception("Arrays must have the same shape. Found {} and {}".format(pred.shape, ref.shape))
    return np.sum(np.minimum(pred, ref))/np.sum(np.maximum(pred, ref))

def contRecall(pred,ref) -> float:
    """ Continuous recall """
    if pred.shape != ref.shape:
        raise Exception("Arrays must have the same shape. Found {} and {}".format(pred.shape, ref.shape))
    return np.sum(np.minimum(pred, ref))/np.sum(ref[:])

def contPrecision(pred,ref) -> float:
    """ Continuous precision """
    if pred.shape != ref.shape:
        raise Exception("Arrays must have the same shape. Found {} and {}".format(pred.shape, ref.shape))
    return np.sum(np.minimum(pred, ref))/np.sum(pred[:])

def fscore(_precision, _recall, beta):
    if all([_precision==0, _recall==0]):
        return np.nan
    else:
        return (1+beta*beta)*_precision*_recall/((beta*beta*_precision)+_recall)

# Generate two arrays, reference and modelled, with continuous, ratio-scale, non-negative measurements
N=5
example = {
    'reference':np.array([[np.round(random.random(),2) for i in range(N)] for j in range(N)])*10,
    'modelled':np.array([[np.round(random.random(),2) for i in range(N)] for j in range(N)])*10
    }

# Plot reference and modelled arrays
fig, axs = plt.subplots(1, 2, dpi=300)
for a, data in enumerate(example):
    axs[a].imshow(example[data], cmap='Blues',vmax=10,interpolation='nearest')
    axs[a].set_title(data+' data')

    ids = [(x,y) for x in range(N) for y in range(N)]
    for (x,y) in ids:
        text = axs[a].text(x,y, np.round(example[data][y,x],2),
                           ha="center", va="center")
    # remove the x and y ticks
    axs[a].set_xticks([])
    axs[a].set_yticks([])
plt.show()

#### Compare magnitude estimations in the reference and modelled arrays    
# Print measures of error
print('ME: %.3f'%ME(example['modelled'], example['reference']))
print('MAE: %.3f'%MAE(example['modelled'], example['reference']))
# Print measures of association
print('r: %.3f'%rho(example['modelled'], example['reference']))
print('Slope: %.3f'%Slope(example['modelled'], example['reference']))
# Print measures of agreement
_contJaccard = contJaccard(example['modelled'], example['reference'])
_contPrecision = contPrecision(example['modelled'], example['reference'])
_contRecall = contRecall(example['modelled'], example['reference'])
print('cont. Jaccard: %.3f'%_contJaccard)
print('cont. Precision: %.3f'%_contPrecision)
print('cont. Recall: %.3f'%_contRecall)
print('cont. F1-score: %.3f'%fscore(_contPrecision, _contRecall, beta=1))






