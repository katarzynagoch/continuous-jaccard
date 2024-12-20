# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 08:23:10 2024

@author: Johannes H. Uhl, European Commission, Joint Research Centre (JRC), Ispra, Italy
"""

try: import gdal
except: from osgeo import gdal
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.stats
import itertools
import matplotlib
from sklearn import preprocessing
matplotlib.rcParams['font.sans-serif'] = 'Arial'
matplotlib.rcParams['font.family'] = "sans-serif"

##### read built-up surface data #####
intif = r'aoi2_Brazil_REF_10.tif'
outdir = r'.\results'
if not os.path.exists(outdir):
    os.makedirs(outdir)

##### function to produce synthetic densities drawn from a skewed normal distribution #####
def randn_skew_fast(N, alpha=0.0, loc=0.0, scale=1.0):
    # generate sample from skewed normal distribution
    # https://stackoverflow.com/questions/36200913/generate-n-random-numbers-from-a-skew-normal-distribution-using-numpy
    sigma = alpha / np.sqrt(1.0 + alpha**2) 
    u0 = np.random.randn(N)
    v = np.random.randn(N)
    u1 = (sigma*u0 + np.sqrt(1.0 - sigma**2)*v) * scale
    u1[u0 < 0] *= -1
    u1 = u1 + loc
    return u1

##### functions to compute categorical and continuous agreement measures #####

def contingency_table(ref, pred):
    # True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
    TP_arr = np.logical_and(ref == 1, pred == 1) # add weights by built-up area : sum by 
    TP = np.sum(TP_arr)
    # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
    TN_arr = np.logical_and(ref == 0, pred == 0)
    TN = np.sum(TN_arr)
    # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
    FP_arr = np.logical_and(ref == 0, pred == 1) # use this as a mask on continous values (sum the built-up area)
    FP = np.sum(FP_arr)
    # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
    FN_arr = np.logical_and(ref == 1, pred == 0)
    FN = np.sum(FN_arr)             
    del TP_arr, TN_arr, FP_arr, FN_arr   
    return TP, TN, FP, FN

def precision(TP, FP):
    return TP/(TP+FP)

def recall(TP, FN):
    return (TP)/(TP+FN)  

def IoU(TP, FP, FN):
    return TP / (TP + FP + FN)

def RMSE(pred,ref) -> float:
    """ Root Mean Square Error """
    if pred.shape != ref.shape:
        raise Exception("Arrays must have the same shape. Found {} and {}".format(pred.shape, ref.shape))
    return np.sqrt(np.mean((pred-ref)**2))

def MAE(pred,ref) -> float:
    """ Mean Absolute Error """
    if pred.shape != ref.shape:
        raise Exception("Arrays must have the same shape. Found {} and {}".format(pred.shape, ref.shape))
    return np.mean(np.absolute(np.subtract(pred,ref)))

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

def MAPE_V2(pred,ref): #https://ideas.repec.org/a/for/ijafaa/y2007i6p40-43.html
    pred=np.round(pred,4)
    ref = np.round(ref,4)    
    pred_mask = np.where(ref>0, pred, np.nan)
    ref_mask = np.where(ref>0, ref, np.nan)
    mae = np.nanmean(np.absolute(np.subtract(pred_mask,ref_mask)))
    mean = np.nanmean(ref_mask)
    mapeV2 = np.divide(mae,mean)*100 # [%]
    return mapeV2#np.mean(np.abs((ref - pred) / ref)) 

def ME(pred,ref): 
    return np.nanmean(pred)-np.nanmean(ref)
    
def MEDAPE(pred,ref) -> float: # ADD MEDAPE
    """ Median Absolute Percentage Error """
    if pred.shape != ref.shape:
        raise Exception("Arrays must have the same shape. Found {} and {}".format(pred.shape, ref.shape))
    pred=np.round(pred,4)
    ref = np.round(ref,4)    
    pred_mask = np.where(ref>0, pred, np.nan)
    ref_mask = np.where(ref>0, ref, np.nan)
    medape = np.nanmedian(np.absolute(np.divide(np.subtract(pred_mask,ref_mask),ref_mask)))*100 # [%]
    return medape

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
    """ Continuous recall """
    if pred.shape != ref.shape:
        raise Exception("Arrays must have the same shape. Found {} and {}".format(pred.shape, ref.shape))
    return np.sum(np.minimum(pred, ref))/np.sum(ref[:])

def contPrecision(pred,ref) -> float:
    """ Continuous precision """
    if pred.shape != ref.shape:
        raise Exception("Arrays must have the same shape. Found {} and {}".format(pred.shape, ref.shape))
    return np.sum(np.minimum(pred, ref))/np.sum(pred[:])

def fscore(_precision, _recall):
    return 2*_precision*_recall/(_precision+_recall) 

##### function that calculates all agreement measures for given data #####

def calc_acc_met(ref_bin,pred_bin,ref_cont,pred_cont,do_balanced=False):    
    
    if do_balanced:
        TP, TN, FP, FN = contingency_table(ref_bin,pred_bin)
        ref_bin = ref_bin.flatten()
        pred_bin = pred_bin.flatten()
        ref_cont = ref_cont.flatten()
        pred_cont = pred_cont.flatten()
        minsize = min([TP,FP,FN])
        tp_idxs = np.where(np.logical_and(ref_bin==1,pred_bin==1))[0]
        fp_idxs = np.where(np.logical_and(ref_bin==0,pred_bin==1))[0]
        fn_idxs = np.where(np.logical_and(ref_bin==1,pred_bin==0))[0]
        tp_idxs_sample = tp_idxs[np.random.randint(0,tp_idxs.shape[0],minsize)]
        fp_idxs_sample = fp_idxs[np.random.randint(0,fp_idxs.shape[0],minsize)]
        fn_idxs_sample = fn_idxs[np.random.randint(0,fn_idxs.shape[0],minsize)]
        idx_combined = np.concatenate([tp_idxs_sample,fp_idxs_sample,fn_idxs_sample])
        ref_bin=ref_bin[idx_combined]
        pred_bin=pred_bin[idx_combined]
        ref_cont=ref_cont[idx_combined]
        pred_cont=pred_cont[idx_combined]  
    TP, TN, FP, FN = contingency_table(ref_bin,pred_bin)
    precision_ = precision(TP, FP) 
    recall_ = recall(TP, FN)
    iou_ = IoU(TP, FP, FN)
    fscore_ = fscore(precision_, recall_)
    RMSE_=RMSE(pred_cont,ref_cont) 
    MAE_=MAE(pred_cont,ref_cont)
    ME_=ME(pred_cont,ref_cont)
    MAPE_=MAPE(pred_cont,ref_cont) 
    MAPEV2_=MAPE_V2(pred_cont,ref_cont) 
    MEDAPE_=MEDAPE(pred_cont,ref_cont) 
    rho_=rho(pred_cont,ref_cont) 
    contJaccard_=contJaccard(pred_cont,ref_cont) 
    contRecall_=contRecall(pred_cont,ref_cont) 
    contPrecision_=contPrecision(pred_cont,ref_cont) 
    contF1=fscore(contPrecision_,contRecall_) 
    return [precision_,recall_,iou_,fscore_,RMSE_,MAE_,ME_,MAPE_,MAPEV2_,MEDAPE_,rho_,contJaccard_,contRecall_,contPrecision_,contF1,TP, TN, FP, FN]

##### read subset of the data to be used as baseline data #####

inarr_ref = gdal.Open(intif).ReadAsArray()[250:850,200:800] # read subset if the data scene
inarr_mod = inarr_ref.copy()

##### produce fake reference data by setting data to zero in some region (injecting false positives), and implanting the removed data in some other, empty region (injecting false negatives) #####

ref_fake = inarr_ref.copy()
ref_fake[475:,:200]=0
implant1 = np.rot90(inarr_ref[475:,:200],2)
ref_fake[:implant1.shape[0],:implant1.shape[1]] = implant1
implant2 = inarr_ref[150:250,300:400]
ref_fake[:implant2.shape[0],ref_fake.shape[1]-implant2.shape[1]:] = implant2

##### produce categorical data by thresholding the continuous data #####

cutoff = 0
ref_bin = ref_fake.copy()
ref_bin[ref_bin>cutoff]=1
mod_bin = inarr_mod.copy()
mod_bin[mod_bin>cutoff]=1
categ_agrmnt_arr = np.zeros(mod_bin.shape)
categ_agrmnt_arr[np.logical_and(ref_bin==0,mod_bin==1)]=1 #fp
categ_agrmnt_arr[np.logical_and(ref_bin==1,mod_bin==1)]=2 #tp
categ_agrmnt_arr[np.logical_and(ref_bin==1,mod_bin==0)]=3 #fn

##### plot agreement categories with fake ref and modelled data
 
fig,axs=plt.subplots(1,3,sharex=True,sharey=True,figsize=(8,4))
ax=axs[0]
ax.imshow(categ_agrmnt_arr,cmap = 'nipy_spectral',interpolation='none')
ax.set_title('categ. agreement map')
ax=axs[1]
ax.imshow(inarr_mod,cmap='turbo',vmin=1,vmax=100)
ax.set_title('modelled')
ax=axs[2]
ax.imshow(ref_fake,cmap='turbo',vmin=1,vmax=100)
ax.set_title('ref')
plt.suptitle('Synthetic reference and test data')
plt.show()

##### read data into dataframe and prepare #####

idxs = np.indices(ref_fake.shape)
df = pd.DataFrame()    
df['cat'] = categ_agrmnt_arr.flatten()
df['density_ref'] = ref_fake.flatten()
df['density_pred'] = inarr_mod.flatten()
df['idx0'] = idxs[0].flatten()
df['idx1'] = idxs[1].flatten()
df = df.sort_values(by='cat').reset_index()
df_no_tn = df[df.cat>0]##### remove true negatives 
df_no_tn['idx_seq'] = np.arange(len(df_no_tn))
fp_tp_dens_pred = df_no_tn[df_no_tn.cat.isin([1,2])]['density_pred'].values
tp_fn_dens_pred = df_no_tn[df_no_tn.cat.isin([2,3])]['density_ref'].values

##### generate and plot expected distribution (baseline case: high density in TP low density in FP and FN domains):

num_samples1 = len(fp_tp_dens_pred)
alpha_skew1 = -3
num_samples2 = len(tp_fn_dens_pred)
alpha_skew2 = 3
fig,ax=plt.subplots() 
p1 = randn_skew_fast(num_samples1, alpha_skew1,loc=0.5)
kde1 = sns.distplot(p1,ax=ax,color='blue',kde_kws={'linestyle':'--'},bins=75)
p2 = randn_skew_fast(num_samples2, alpha_skew2,loc=-0.5)
kde2 = sns.distplot(p2,ax=ax,color='orange',bins=75)
plt.title('Case 1: expected')
plt.show()
fig.savefig(outdir+os.sep+'000_synth_distributions.png',dpi=300)

##### sample from distributions and write values back to density surfaces:
    
pdf1 = kde1.lines[0].get_data()[1]
pdf1 = pdf1 + np.min(pdf1) + 1
pdf1 = 100*(pdf1 / np.max(pdf1))
pdf1_densified = np.interp(np.arange(num_samples1), np.linspace(0,num_samples1,num = len(pdf1)), pdf1)
pdf2 = kde2.lines[0].get_data()[1]
pdf2 = pdf2 + np.min(pdf2) + 1
pdf2 = 100*(pdf2 / np.max(pdf2))
pdf2_densified = np.interp(np.arange(num_samples2), np.linspace(0,num_samples2,num = len(pdf2)), pdf2)
df_no_tn['case1_dens_pred']=0
df_no_tn.loc[df_no_tn.cat.isin([1,2]),['case1_dens_pred']] = pdf1_densified
df_no_tn['case1_dens_ref']=0
df_no_tn.loc[df_no_tn.cat.isin([2,3]),['case1_dens_ref']] = pdf2_densified

ingest_values_pred = df_no_tn.case1_dens_pred.values
ingest_values_ref = df_no_tn.case1_dens_ref.values
dens_arr_pred = np.zeros(ref_fake.shape)
dens_arr_pred[df_no_tn.idx0,df_no_tn.idx1] = ingest_values_pred
dens_arr_ref = np.zeros(ref_fake.shape)
dens_arr_ref[df_no_tn.idx0,df_no_tn.idx1] = ingest_values_ref

##### now we design the different scenarios #####

accmetrics=[]
factors=[2]# the factor by which we increase the densitiees per agreement category
boxplot_data=[]
for factor in factors:
    lst = list(itertools.product([0, 1], repeat=4))
    modelnumber=0
    for combi in lst: 
        ##### for each domain-state combination 
        ##### (state=increase or not increase, 
        ##### domain=tp_ref,tp_mod,fp_mod,fn_ref)
        ##### increase the values or not        
        modelnumber+=1
        fp_factor,tp_factor_pred,tp_factor_ref,fn_factor = combi
    
        if fp_factor==1:
            fp_factor=factor
        if tp_factor_pred==1:
            tp_factor_pred=factor
        if tp_factor_ref==1:
            tp_factor_ref=factor
        if fn_factor==1:
            fn_factor=factor
            
        if fp_factor==0:
            fp_factor=1
        if tp_factor_pred==0:
            tp_factor_pred=1
        if tp_factor_ref==0:
            tp_factor_ref=1
        if fn_factor==0:
            fn_factor=1        
            
        curr_df_no_tn = df_no_tn.copy()

        ####### FP
        modvalues = curr_df_no_tn.loc[curr_df_no_tn.cat.isin([1])].case1_dens_pred.values*fp_factor
        np.random.shuffle(modvalues)
        curr_df_no_tn.loc[curr_df_no_tn.cat.isin([1]),['case1_dens_pred']] = modvalues
        
        ####### TP
        modvalues = curr_df_no_tn.loc[curr_df_no_tn.cat.isin([2])].case1_dens_pred.values*tp_factor_pred
        np.random.shuffle(modvalues)
        curr_df_no_tn.loc[curr_df_no_tn.cat.isin([2]),['case1_dens_pred']] = modvalues
        modvalues = curr_df_no_tn.loc[curr_df_no_tn.cat.isin([2])].case1_dens_ref.values*tp_factor_ref     
        np.random.shuffle(modvalues)
        curr_df_no_tn.loc[curr_df_no_tn.cat.isin([2]),['case1_dens_ref']] = modvalues
        
        ####### FN
        modvalues = curr_df_no_tn.loc[curr_df_no_tn.cat.isin([3])].case1_dens_ref.values*fn_factor
        np.random.shuffle(modvalues)
        curr_df_no_tn.loc[curr_df_no_tn.cat.isin([3]),['case1_dens_ref']] = modvalues
        
        ingest_values_pred = curr_df_no_tn.case1_dens_pred.values
        ingest_values_ref = curr_df_no_tn.case1_dens_ref.values
        dens_arr_pred = np.zeros(ref_fake.shape)
        dens_arr_pred[curr_df_no_tn.idx0,curr_df_no_tn.idx1] = ingest_values_pred
        dens_arr_ref = np.zeros(ref_fake.shape)
        dens_arr_ref[curr_df_no_tn.idx0,curr_df_no_tn.idx1] = ingest_values_ref
        
        ##### convert a long-form data frame from the modified data, for the boxplot #####

        df_longform1=pd.DataFrame()        
        df_longform1['value'] = curr_df_no_tn[curr_df_no_tn.cat==1].case1_dens_pred.values
        df_longform1['domain']='1fp_pred'
        mean_density_fp = df_longform1.value.mean() 
        sum_density_fp = df_longform1.value.sum() 
        df_longform2=pd.DataFrame()        
        df_longform2['value'] = curr_df_no_tn[curr_df_no_tn.cat==2].case1_dens_pred.values
        df_longform2['domain']='2tp_pred'        
        mean_density_tp_pred = df_longform2.value.mean()  
        sum_density_tp_pred = df_longform2.value.sum()        
        df_longform3=pd.DataFrame()        
        df_longform3['value'] = curr_df_no_tn[curr_df_no_tn.cat==2].case1_dens_ref.values
        df_longform3['domain']='3tp_ref'  
        mean_density_tp_ref = df_longform3.value.mean()   
        sum_density_tp_ref = df_longform3.value.sum()        
        df_longform4=pd.DataFrame()        
        df_longform4['value'] = curr_df_no_tn[curr_df_no_tn.cat==3].case1_dens_ref.values
        df_longform4['domain']='4fn_ref'  
        mean_density_fn = df_longform4.value.mean()  
        sum_density_fn = df_longform4.value.sum()                              
        df_longform = pd.concat([df_longform1,df_longform2,df_longform3,df_longform4])
        
        ##### calculate and save agreement metrics #####

        accmetrics_current = calc_acc_met(ref_bin,mod_bin,dens_arr_ref,dens_arr_pred,do_balanced=False)
        precision_,recall_,iou_,fscore_,RMSE_,MAE_,ME_,MAPE_,MAPEV2_,MEDAPE_,rho_,contJaccard_,contRecall_,contPrecision_,contF1,TP, TN, FP, FN = accmetrics_current        
        boxplot_data.append([contJaccard_,df_longform.copy(),mean_density_fp,mean_density_tp_pred,mean_density_tp_ref,mean_density_fn])        
        accmetrics.append([factor,fp_factor,tp_factor_pred,tp_factor_ref,fn_factor] + [mean_density_fp,mean_density_tp_pred,mean_density_tp_ref,mean_density_fn] + [sum_density_fp,sum_density_tp_pred,sum_density_tp_ref,sum_density_fn] + accmetrics_current)
        print(modelnumber,accmetrics_current)
        
        ##### plot the maps of density distributions for modelled and reference data, for the current scenario: #####
        
        fig,axs=plt.subplots(1,2,figsize=(12,5),sharey=True)      
        ax=axs[0]
        ax.imshow(dens_arr_pred,cmap='magma',vmax=100)
        ax.set_title('Modelled density')
        ax=axs[1]
        ax.imshow(dens_arr_ref,cmap='magma',vmax=100)
        ax.set_title('Reference density')
        plt.suptitle('Scenario %s' %(modelnumber))        
        plt.show()        
        fig.savefig(outdir+os.sep+'2x1_panel_boxplots_%s.png' %modelnumber,dpi=150)            
 

##### export agreement metrics to csv #####        
accmetricsdf = pd.DataFrame(accmetrics)    
accmetricsdf.columns=['factor','fp_factor','tp_factor_pred','tp_factor_ref','fn_factor','mean_density_fp','mean_density_tp_pred','mean_density_tp_ref','mean_density_fn','sum_density_fp','sum_density_tp_pred','sum_density_tp_ref','sum_density_fn','precision_cat','recall_cat','iou_cat','fscore_cat','RMSE_',
                      'MAE_','ME_','MAPE_','MAPEV2_','MEDAPE_','rho_','contJaccard_','contRecall_','contPrecision_','contF1','TP', 'TN', 'FP', 'FN']    
accmetricsdf.to_csv(outdir+os.sep+'scenarios_accmeas.csv',index=False)    

##########################################################################################################################################

##### visualize results: create scatterplot and cross-correlation matrix #####

# code adjusted from https://github.com/johannesuhl/scatterplot_matrix

datadf=accmetricsdf
variables=np.array(['RMSE_','MAE_','ME_','MAPE_','MAPEV2_','rho_','contJaccard_','contRecall_','contPrecision_','contF1'])
variables_names = ['RMSE','MAE','ME','MAPE','wMAPE','r','cont. Jaccard', 'cont. Recall', 'cont. Precision', 'cont. F1']
namedict=dict(zip(variables,variables_names))
datadf=datadf[variables]

### column for color coding:
colorcode_scatter = 'contJaccard_'

exclude_colorcoded=False ### will exclude the variable used for colorcoding from the scatterplot / correlation matrix.
use_ranks=True ### generate a matrix of QQ plots rather than scatterplots
transform_to_01 = False ### transforms each column to [0,1]
standardize = False ### standardize each column
fs=9 # font size

matplotlib.rcParams['font.sans-serif'] = "Arial"
matplotlib.rcParams['font.family'] = "sans-serif"
matplotlib.rcParams['font.size'] = fs 

### cross correlation matrix:
colorscheme = 'coolwarm'
cmap_corr = plt.cm.get_cmap(colorscheme)
cmapvals_corr = np.arange(0.0,1.0,1000)

crosscorrmat = np.empty((variables.shape[0],variables.shape[0]))
var1count=0
sorted_vars=[]
for var1 in datadf.columns:
    sorted_vars.append(var1)
    var2count=0
    for var2 in datadf.columns:
        if use_ranks:
            crosscorr = scipy.stats.spearmanr(datadf[var1].values,datadf[var2].values)[0]
        else:
            crosscorr = scipy.stats.pearsonr(datadf[var1].values,datadf[var2].values)[0]
            
        crosscorrmat[var1count,var2count] = crosscorr                                    
        var2count+=1                                   
    var1count+=1    
crosscorrmat = np.nan_to_num(crosscorrmat)

if transform_to_01:
    min_max_scaler = preprocessing.MinMaxScaler()
    for col in variables:
        x = datadf[[col]].values.astype(float)
        datadf[col] = min_max_scaler.fit_transform(x) 

if standardize:
    min_max_scaler = preprocessing.StandardScaler()
    for col in variables:
        x = datadf[[col]].values.astype(float)
        datadf[col] = min_max_scaler.fit_transform(x) 
    
if exclude_colorcoded:
    if use_ranks:
        for col in datadf[variables]:
            datadf[col]=datadf[col].rank(pct=True)
    indarr = datadf[variables].drop(labels=[colorcode_scatter],axis=1).values
else:
    if use_ranks:
        for col in datadf[variables]:
            datadf[col]=datadf[col].rank(pct=True)
    indarr = datadf[variables].values   
    
cmap1 = matplotlib.cm.turbo #colorcoding for variable
cmapvals1 = 1-((np.max(indarr[:,np.argwhere(variables==colorcode_scatter)[0][0]])-indarr[:,np.argwhere(variables==colorcode_scatter)[0][0]])/float(np.max(indarr[:,np.argwhere(variables==colorcode_scatter)[0][0]])-np.min(indarr[:,np.argwhere(variables==colorcode_scatter)[0][0]]))) 
currcols1 = cmap1(cmapvals1) 

if exclude_colorcoded:
    variables=np.array([xx for xx in variables if xx not in [colorcode_scatter]])

fig, axes = plt.subplots(len(variables), len(variables), figsize=(7,7),sharex=True,sharey=True)
fig.subplots_adjust(hspace=0.05)
fig.subplots_adjust(wspace=0.05)

var1count=0
for var1 in variables:
    var2count=0
    for var2 in variables:
        ax = axes[var1count,var2count]
        if var1count==var2count: #main diagonal
            ax = axes[var1count,var2count]
            ax.patch.set_facecolor('lightgrey')        
            if var1count==0 and var2count==0:                
                ax.set_title(namedict[var1],fontsize=fs,rotation=45)
                ax.set_ylabel(namedict[var2] ,fontsize=fs,rotation=45,ha='right')             
        else:
            if not var1count>var2count:  ## upper triangle, colorcode by first variable ####################
                print (var1count,var2count)
                print (var1,var2)
                var1vals = indarr[:,np.argwhere(variables==var1)[0][0]]
                var2vals =indarr[:,np.argwhere(variables==var2)[0][0]]
                ax.patch.set_facecolor('white')
                im = ax.scatter(x=var2vals, y=var1vals, s=7, color = currcols1, alpha=1)
                #ax.set_xlim([0,1])
                #ax.set_ylim([0,1]) 
                #ax.get_xaxis().set_ticks([0,1])
                #ax.get_yaxis().set_ticks([0.1])                     
                if var1count==0:
                    ax.set_title(namedict[var2],fontsize=fs,rotation=45)
                    if var2count==0: 
                        ax.set_ylabel(namedict[var1],fontsize=fs,rotation=45)                                                
            else: ## lower triangle, show cross correlation ####################
                ax = axes[var1count,var2count]
                crosscor = crosscorrmat[var1count,var2count]
                crosscor_scaled = (1+crosscor)/2.0
                ax.patch.set_facecolor(cmap_corr(crosscor_scaled))
                ax.annotate("%.2f" % crosscor,(np.mean(axes[var2count,var1count].get_xlim()),np.mean(axes[var2count,var1count].get_ylim())),ha='center', va = 'center',fontsize=fs)
                ax.set_xlim(axes[var2count,var1count].get_xlim())
                ax.set_ylim(axes[var2count,var1count].get_ylim())                 
                if var2count==0:
                    ax.set_ylabel(namedict[var1],fontsize=fs,rotation=45,ha='right')  
                    if var1count==0:
                        ax.set_title(namedict[var2],fontsize=fs,rotation=45)                    
        var2count+=1                                   
    var1count+=1
plt.xticks([], []) # remove ticks
plt.yticks([], [])
fig.savefig(outdir+os.sep+'scatterplot_matrix_w_crosscorr.png', dpi=1200)  

##### visualize results: boxplots of density distributions per domain, for each scenario: ##### 

boxplot_datadf= pd.DataFrame(boxplot_data,columns=['acc','data','mean_density_fp','mean_density_tp_pred','mean_density_tp_ref','mean_density_fn'])
# boxplot_datadf = boxplot_datadf.sort_values(by='acc',ascending=False).reset_index()
boxplot_datadf['counter']=np.arange(1,len(boxplot_datadf)+1)
fig,axs = plt.subplots(2,int(len(boxplot_data)/2),sharex=True,sharey=True,figsize=(12,4))
counter=0
plotcol=0
for i,row in boxplot_datadf.iterrows():
    if counter==8:
        plotcol=1
        counter=0
    print(row.acc)
    currdf = row.data
    ax=axs[plotcol,counter]
    my_pal = {"1fp_pred": "darkturquoise", "2tp_pred": "yellow", "3tp_ref":"yellow", "4fn_ref":"lightgrey"}
    bx = sns.boxplot(x=currdf.domain,y=currdf.value,ax=ax,palette=my_pal)
    ax.set_ylim([-5,220])
    
    if counter==0:
        ax.set_ylabel('Density')
    else:
        ax.set_ylabel('')
    if plotcol==0:
        ax.set_xlabel('')
    else:
        ax.set_xlabel('Domain')
        ax.set_xticklabels(['FP (mod.)','TP (mod.)','TP (ref.)','FN (ref.)'],rotation=45)
    ax.set_title(row.counter)  
    counter+=1
plt.show()   
fig.tight_layout() 
fig.savefig(outdir+os.sep+'boxplots 16 scenarios 2m.png' ,dpi=300)

##### visualize results: barcharts of mean density per domain and scenario #####
    
accmetricsdf = accmetricsdf.sort_values(by='contJaccard_',ascending=False).reset_index()
fig,axs = plt.subplots(len(boxplot_data),1,sharex=True,sharey=True,figsize=(6,20))
counter=0
for i,row in accmetricsdf.iterrows():
    print(row.contJaccard_)
    ax=axs[counter]
    means = row[['mean_density_fp','mean_density_tp_pred','mean_density_tp_ref','mean_density_fn']].values
    ax.bar(x=[1,2,3,4],height  =means )
    ax.set_ylim([-5,210])

    if counter==0:
        ax.set_ylabel('Mean density')     
    counter+=1
plt.show()  
fig.savefig(outdir+os.sep+'barcharts 16 scenarios 2m.png' ,dpi=150)
