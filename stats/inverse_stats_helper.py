#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 09:58:02 2021

@author: leupinv
"""
## libraries
from statsmodels.stats.multitest import fdrcorrection_twostage
import numpy as np
import mne
from scipy import stats
from mne.stats import fdr_correction, f_mway_rm
import matplotlib.pyplot as plt
import base.base_constants as bs
from mne.stats import (
    spatio_temporal_cluster_1samp_test, summarize_clusters_stc, f_threshold_mway_rm,
    spatio_temporal_cluster_test, permutation_cluster_test, ttest_1samp_no_p)
import os
import seaborn as sns
import pandas as pd

VERTICES_SINGLE = 2562

## General use functions


def filter_list(list_: list, value: str) -> list:
    '''


    Parameters
    ----------
    list_ : list
        DESCRIPTION: a list containing strings
    value : str
        DESCRIPTION. The string to identify

    Returns
    -------
    list
        DESCRIPTION. list of str containing the str you want to find

    '''
    filter_list = []
    for x in list_:
        whoe_dir = x.replace("\\", '/').split('/')
        name = whoe_dir[-1]
        name = name.split('.')[0].split('-')[0]

        if value == name:
            filter_list.append(x)

    return filter_list


## T-test functions

def get_list_cond(stc_files: list, cond: str,g_list=None):
    list_stc = []
    if g_list is not None:
        if '_o_' in cond:
            print('this is oral')
            for g in g_list:
                finder = f'{g}_{cond}'
            filt_Val = filter_list(stc_files, finder)
            print(filt_Val)

            stc = mne.read_source_estimate(filt_Val[0][:-7])
            list_stc.append(stc)
        else:
            print('this is nasal')
            for g in g_list:
                finder = f'{g}_n_tsk_{cond}'
                filt_Val = filter_list(stc_files, finder)
                print(filt_Val)

                stc = mne.read_source_estimate(filt_Val[0][:-7])
                list_stc.append(stc)
        return list_stc

    if '_o_' in cond:
        print('this is oral')
        for g in bs.G_N:
            finder = f'{g}_{cond}'
            filt_Val = filter_list(stc_files, finder)
            print(filt_Val)

            stc = mne.read_source_estimate(filt_Val[0][:-7])
            list_stc.append(stc)
    else:
        print('this is nasal')
        for g in bs.G_N:
            finder = f'{g}_n_tsk_{cond}'
            filt_Val = filter_list(stc_files, finder)
            print(filt_Val)

            stc = mne.read_source_estimate(filt_Val[0][:-7])
            list_stc.append(stc)
    return list_stc


def get_DF(evoked, crop_value=None, g_excl=None):
    '''


    Parameters
    ----------
    evoked : TYPE
        DESCRIPTION. list containing 2 evokeds mne object
    crop_value : TYPE, optional
        DESCRIPTION. The default is None. touple cointaining time limits
    g_excl : TYPE, optional
        DESCRIPTION. The default is None. list containing subjects you want to exclude

    Returns
    -------
    X : TYPE
        DESCRIPTION. formatted subject data

    '''
    if crop_value != None:
        data_crop = evoked[0].crop(crop_value[0], crop_value[1])
        data_shape = data_crop.data.shape
        subj_len = len(evoked)

    else:
        data_shape = evoked[0].data.shape
        subj_len = len(evoked)
    # if g_excl!= None:
    # subj_len=len(evoked)-len(g_excl)
    # evoked=[ev for ev in evoked if not any(g in ev.comment for g in g_excl) ]

    X = np.empty((subj_len, data_shape[0], data_shape[1]))

    if crop_value != None:
        for idx, ev in enumerate(evoked):
            X[idx, :, :] = ev.crop(crop_value[0], crop_value[1]).data

        print(np.shape(X))

    else:
        for idx, ev in enumerate(evoked):
            X[idx, :, :] = ev.data

    return X


def len_check(evoked, length):
    truth_seeker = [len(x) == length for x in evoked]
    if all(truth_seeker) == False:
        raise Exception('Unequal number of subjects')


def extract_lbl(stc, lab_range=(0.09, 0.12), src='fsaverage', smooth=False):
    tmin = lab_range[0]
    tmax = lab_range[1]
    stc_mean = stc.copy().crop(tmin, tmax).mean()

    label = mne.stc_to_label(stc_mean, src=src, smooth=smooth)

    return label


def visualize_label(target_label, stc):
    vx = target_label.get_vertices_used(np.arange(2562))
    if 'rh' in target_label.name:
        vx = vx+2562
    stc_vx = np.zeros(np.shape(stc.data))
    stc_vx[vx, :] = 1
    stc_vx_plot = mne.SourceEstimate(
        stc_vx, stc.vertices, tmin=-0.1, tstep=stc.tstep)
    stc_vx_plot.plot('fsaverage', hemi='both')


def get_sig_label_points(stc_obj, target_lab, src, smooth=True):
    stc_lab = stc_obj.in_label(target_lab)
    sig_lab = mne.stc_to_label(stc_lab, src=src, smooth=smooth)
    if sig_lab[0] is not None:
        sig_lab[0].name = f'{target_lab.name[:-3]}_sig_lh'
    if sig_lab[1] is not None:
        sig_lab[1].name = f'{target_lab.name[:-3]}_sig_rh'

    sig_lab = [lab for lab in sig_lab if lab is not None]
    return sig_lab[0]


def plot_label_effect(stc_obj, stc_label):

    stc_label_lh = stc_label[0]
    stc_label_rh = stc_label[1]
    stc_mean_label_lh = stc_obj.in_label(stc_label_lh)
    stc_mean_label_rh = stc_obj.in_label(stc_label_rh)
    return stc_mean_label_lh, stc_mean_label_rh


def Ttest_core(X, p_val=0.05, FDR=False):
    # T-test
    out = stats.ttest_1samp(X, 0, axis=0)
    ts = out[0]
    ps = out[1]

   # establish significancy mask (can be FDR or noc)
    if FDR:
        reject_fdr, pval_fdr = fdr_correction(ps)
        #reject_fdr, pval_fdr,_ ,_ = fdrcorrection_twostage(ps)
        print(reject_fdr)
        sig_value = reject_fdr
    else:
        sig_value = ps < p_val
    print(ts)

    mask_ts = ts.copy()
    print(mask_ts)

    mask_ts[sig_value == False] = 0

    return ts, ps, mask_ts


def get_tTest(X, list_stc, FDR=False, plot_times='peaks', averages=None, p_val=.05, report=None,
              crop_value=None, time_smooth=False):
    '''


    Parameters
    ----------
    X : TYPE
        DESCRIPTION. Array containing formatted subject data
    label : TYPE, optional
        DESCRIPTION. The default is None. label to generate report with
    FDR : TYPE, optional
        DESCRIPTION. The default is False so NOC. Weather correct or not witf FDR
    plot_times : TYPE, optional expects list of times or any other keyword accepted by mne
        DESCRIPTION. The default is 'peaks'. what times to plot on the topoplot
    averages : TYPE, optional
        DESCRIPTION. The default is None. how much to average the topoplots
    p_val : TYPE, optional
        DESCRIPTION. P-value, The default is .05.
    report : TYPE, optional
        DESCRIPTION. The default is None. Report object that needs to be passed

    Returns
    -------
    None.sp

    '''

    ts, ps, mask_ts = Ttest_core(X, p_val=p_val, FDR=FDR)

    #plt.figure()
    #plt.imshow(ts)
    #plt.colorbar()
    stc_obj = stc_plot(mask_ts, list_stc, tmin=crop_value[0])
    # generate whole picture
    kwargs = dict(hemi='both', subject='fsaverage',
                  size=(600, 600))
    brain_auto = stc_obj.plot(figure=None, **kwargs)
    # fig_evo=stc.plot_image(mask=sig_value,scalings=1,units='T-value',show_names='auto')
    #
    # generate topoplot
    # fig_topo=evok.plot_topomap(plot_times,outlines='head',scalings=1,units='T-value',average=averages,mask=sig_value)

    # add graphs to report to produce HTML only if label is present
    if FDR:
        corr = 'FDR corrected'
    else:
        corr = 'noc'

    return stc_obj


def stc_plot(stc_effect, list_stc, tmin):
    stc = mne.SourceEstimate(
        stc_effect, list_stc[0].vertices, tmin=tmin, tstep=list_stc[0].tstep)

    return stc


def tTest_ana(stc_files, search_lab, crop_value=None, FDR=False,
              p_val=.05, g_excl=None, report=None, lab_range=(0.09, 0.12), time_smooth=False):
    '''
    Wrapper to run Ttest_ana in one line

    Parameters
    ----------
    evoked : TYPE
        DESCRIPTION. Array containing formatted subject data
    label : TYPE, optional
        DESCRIPTION. The default is None.label to generate report with
    crop_value : TYPE, optional
        DESCRIPTION. The default is None. Touple cointaining time limits
    FDR : TYPE, optional
        DESCRIPTION. The default is False so NOC. Weather correct or not witf FDR
    plot_times : TYPE, optional, expects list of times or any other keyword accepted by mne
        DESCRIPTION. The default is 'peaks'.
    averages : TYPE, optional
        DESCRIPTION. The default is None. How much to average the topoplots
    p_val : TYPE, optional
        DESCRIPTION. P-value, The default is .05.
    g_excl : TYPE, optional
        DESCRIPTION. The default is None.
    report : TYPE, optional
        DESCRIPTION. The default is None. Report object that needs to be passed

    Returns
    -------
    report : TYPE
        DESCRIPTION. report object

    '''

    stc_1 = get_list_cond(stc_files, search_lab[0])
    stc_2 = get_list_cond(stc_files, search_lab[1])

    X_1 = get_DF(stc_1, crop_value=crop_value, g_excl=g_excl)
    X_2 = get_DF(stc_2, crop_value=crop_value, g_excl=g_excl)

    mean_X_1 = np.mean(X_1, axis=0)
    mean_X_2 = np.mean(X_2, axis=0)

    stc_mean_X1 = stc_plot(mean_X_1, list_stc=stc_1, tmin=crop_value[0])
    stc_mean_X2 = stc_plot(mean_X_2, list_stc=stc_2, tmin=crop_value[0])
    kwargs = dict(hemi='both', subject='fsaverage',
                  size=(600, 600))
    #stc_mean_X1.plot(figure=None,**kwargs)
    #stc_mean_X2.plot(figure=None,**kwargs)

    n_rep = len(stc_1)
    len_check([stc_1, stc_2], n_rep)
    if time_smooth:
        X_1 = np.mean(X_1, 2)
        X_2 = np.mean(X_2, 2)

    X = X_1 - X_2
    print(X)
    stc_obj = get_tTest(X, stc_1, FDR=FDR, p_val=p_val,
                        crop_value=crop_value)

    return stc_obj


def get_label_time_course(stc_files, search_lab, stc_label, crop_value, g_excl=None, T_test=False, p_val=0.05, FDR=False):
    fig, axs = plt.subplots(2)
    stc_labels_lh = []
    stc_labels_rh = []
    for condition in search_lab:
        #get conditions
        stc = get_list_cond(stc_files, condition)
    #get DF
        X_1 = get_DF(stc, crop_value=crop_value, g_excl=g_excl)
    #compute grand averages
        mean_X_1 = np.mean(X_1, axis=0)
    #get stc format for grand averages

        stc_mean_X1 = stc_plot(mean_X_1, list_stc=stc, tmin=crop_value[0])
    #isolate only locations in the label
        stc_lh_1, stc_rh_1 = plot_label_effect(stc_mean_X1, stc_label)
        stc_labels_lh.append(stc_lh_1.data.mean(0))
        stc_labels_rh.append(stc_rh_1.data.mean(0))

    #plot time course of the mean across label locations

        axs[0].plot(stc_mean_X1.times, stc_lh_1.data.mean(0),
                    label=condition+'_lh')
        axs[1].plot(stc_mean_X1.times, stc_rh_1.data.mean(0),
                    label=condition+'_rh')

    axs[0].set_title('timecourse of ' + stc_label[0].name)

    axs[1].set_title('timecourse of ' + stc_label[1].name)
    axs[0].legend()
    axs[1].legend()


def t_test_label_time_course(stc_list, search_lab, stc_label, crop_value, g_excl=None, p_val=0.05, FDR=False):
    stc_label_V = stc_label.get_vertices_used(np.arange(VERTICES_SINGLE))

    #get conditions
    stc_1 = stc_list[0]
    stc_2 = stc_list[1]
    #get DF
    X_1 = get_DF(stc_1, crop_value=crop_value, g_excl=g_excl)
    X_2 = get_DF(stc_2, crop_value=crop_value, g_excl=g_excl)

    data = X_1-X_2

    if 'rh' in stc_label.name:
        rh = True
        stc_label_V += VERTICES_SINGLE
        data_lab = data[:, stc_label_V, :]
    else:
        data_lab = data[:, stc_label_V, :]
        rh = False
    print(f'the rh is {rh}')
    print(np.shape(data_lab))
    data_lab = np.mean(data_lab, axis=1)

    ts, ps, mask_ts = Ttest_core(data_lab, p_val=p_val, FDR=FDR)

    time_plot = np.arange(len(ts))/256*1000 - \
        np.abs((crop_value[0]*1000))
    print(np.shape(ts))
    plt.figure()
    plt.plot(time_plot, np.mean(np.mean(
        X_1[:, stc_label_V, :], axis=1), axis=0), label=f'{search_lab[0]}')
    plt.plot(time_plot, np.mean(np.mean(
        X_2[:, stc_label_V, :], axis=1), axis=0), label=f'{search_lab[1]}')
    axes = plt.gca()
    y_min, y_max = axes.get_ylim()
    mask_ts[mask_ts != 0] = y_max
    plt.plot(time_plot, mask_ts, 'k')
    plt.legend()
    #axs_t[0].plot(time_plot[mask_ts_lh !=0],mask_ts_lh[mask_ts_lh!=0],label='lh')

    plt.figure()
    plt.plot(time_plot, ts)


def Anovas_label_time_course(stc_list, search_lab, stc_label, crop_value, effects_labels, g_excl=None, p_val=0.05, FDR=False, factor_levels=[2, 2], effects='A*B', plot_bar=False):
    stc_label_V = stc_label.get_vertices_used(np.arange(VERTICES_SINGLE))

    # get raw X
    X = [get_DF(X, crop_value=crop_value, g_excl=g_excl) for X in stc_list]
    print(np.shape(X))
    # format X
    ## define dimensions
    n_rep = len(stc_list[0])
    n_conditions = len(stc_list)
    n_chan = np.shape(X)[2]
    print(n_chan)
    n_times = np.shape(X)[3]
    print(n_times)

    ## get stc_labels
    if 'rh' in stc_label.name:
        rh = True
        data_lab_list = [np.mean(data[:, stc_label_V+VERTICES_SINGLE, :], axis=1)
                         for data in X]
    else:
        data_lab_list = [np.mean(data[:, stc_label_V, :], axis=1)
                         for data in X]
        rh = False
    print(f'the rh is {rh}')
    print(np.shape(data_lab_list))

    ## reformat data
    data_lab = np.swapaxes(np.asarray(data_lab_list), 1, 0)

    print(np.shape(data_lab))
    # reshape last two dimensions in one mass-univariate observation-vector
    #data_lh = data_lh.reshape(n_rep, n_conditions, n_times)
    #data_rh = data_rh.reshape(n_rep, n_conditions, n_times)

    # Compute Anova
    fvals, pvals = f_mway_rm(
        data_lab, factor_levels, effects=effects, correction=True)

    if FDR:
        sign, fdr = fdr_correction(pvals)
    else:
        sign = pvals < p_val

    # Plot anova
    for effect, sig, effect_label in zip(fvals, sign, effects_labels):

        #time_plot = np.arange(len(effect))/256*1000 + \
        #(crop_value[0]*1000)
        time_plot = np.linspace(
            crop_value[0]*1000, crop_value[1]*1000, len(effect))
        print(len(effect))
        plt.figure()
        plt.plot(time_plot, np.mean(data_lab_list[0], axis=0),
                 label=f'{search_lab[0]}')
        plt.plot(time_plot, np.mean(data_lab_list[1], axis=0),
                 label=f'{search_lab[1]}')
        plt.plot(time_plot, np.mean(data_lab_list[2], axis=0),
                 label=f'{search_lab[2]}')
        plt.plot(time_plot, np.mean(data_lab_list[3], axis=0),
                 label=f'{search_lab[3]}')
        axes = plt.gca()
        y_min, y_max = axes.get_ylim()
        mask_ts_lh = np.zeros(len(effect))
        mask_ts_lh[sig] = y_max

        plt.plot(time_plot, mask_ts_lh, 'k')
        plt.legend()

        plt.title(
            f'Effect of {effect_label} for label: {stc_label.name[:-3]}')
        plt.figure()
        plt.plot(time_plot, effect, label='lh')
    if plot_bar:
        data_lab_list = [np.mean(data, axis=1)for data in data_lab_list]
        return data_lab_list
    else:
        return None


def Anovas_clus_label_time_course(stc_list, search_lab, stc_label, crop_value, effects_labels, g_excl=None, p_val=0.05, FDR=False, factor_levels=[2, 2], effects='A:B', n_perm=50, thresh=None, label=None):
    stc_label_V = stc_label.get_vertices_used(np.arange(VERTICES_SINGLE))

    # get raw X
    X = [get_DF(X, crop_value=crop_value, g_excl=g_excl) for X in stc_list]
    print(np.shape(X))
    # format X
    ## define dimensions
    n_rep = len(stc_list[0])
    n_conditions = len(stc_list)
    n_chan = np.shape(X)[2]
    print(n_chan)
    n_times = np.shape(X)[3]
    print(n_times)

    ## get stc_labels
    if 'lh' in stc_label.name:
        data_list = [np.mean(data[:, stc_label_V, :], axis=1)
                     for data in X]
    elif 'rh' in stc_label.name:
        data_list = [np.mean(data[:, stc_label_V+VERTICES_SINGLE, :], axis=1)
                     for data in X]
    else:
        raise ValueError('problem with label, no rh nor lh')
    print(np.shape(data_list))

    if thresh == None:
        f_threshold = f_threshold_mway_rm(
            n_rep, factor_levels, effects, pvalue=p_val)
        print(f_threshold)
    elif thresh == 'TFCE':
        f_threshold = dict(start=0, step=0.2)

    def stat_fun_anov(*args):
        return f_mway_rm(np.swapaxes(args, 1, 0), factor_levels=factor_levels, effects=effects, return_pvals=False)[0]

    # Compute Anova
    F_obs, clusters, clust_p, H0 = clus_return =\
        permutation_cluster_test(
            data_list, threshold=f_threshold, stat_fun=stat_fun_anov, n_permutations=n_perm, n_jobs=-1, out_type='mask')

    clus_thresh = 0.11
    time_plot = np.linspace(crop_value[0]*1000, crop_value[1]*1000, len(F_obs))

    for clu, clu_p in zip(clusters, clust_p):
        if clu_p < clus_thresh:
            plt.figure()
            plt.plot(time_plot, np.mean(data_list[0], axis=0),
                     label=f'{search_lab[0]}')
            plt.plot(time_plot, np.mean(data_list[1], axis=0),
                     label=f'{search_lab[1]}')
            plt.plot(time_plot, np.mean(data_list[2], axis=0),
                     label=f'{search_lab[2]}')
            plt.plot(time_plot, np.mean(data_list[3], axis=0),
                     label=f'{search_lab[3]}')

            axes = plt.gca()
            y_min, y_max = axes.get_ylim()
            mask_ts_lh = np.zeros(len(F_obs))
            mask_ts_lh[clu] = y_max
            plt.plot(time_plot, mask_ts_lh, 'k')
            plt.legend()

            plt.title(
                f'Effect of {effects_labels} for label: {stc_label.name[:-3]}')

    #F_obs_plot = plot_stc_clu(clu)

    #plt.plot(F_obs_plot)
    return clus_return, data_list
    # for effect_lh, sig_lh, effect_rh, sig_rh, effect_label in zip(fvals_lh, sign_lh, fvals_rh, sign_rh, effects_labels):
#
    # time_plot = np.arange(len(effect_lh))/256*1000 - \
    # np.abs((crop_value[0]*1000))
#
    # print(len(effect_lh))
    # fig_t, axs_t = plt.subplots(2)
    # axs_t[0].plot(time_plot, np.mean(data_lh_list[0], axis=0),
    # label=f'{search_lab[0]}_lh')
    # axs_t[0].plot(time_plot, np.mean(data_lh_list[1], axis=0),
    # label=f'{search_lab[1]}_lh')
    # axs_t[0].plot(time_plot, np.mean(data_lh_list[2], axis=0),
    # label=f'{search_lab[2]}_lh')
    # axs_t[0].plot(time_plot, np.mean(data_lh_list[3], axis=0),
    # label=f'{search_lab[3]}_lh')
    # y_min, y_max = axs_t[0].get_ylim()
    # mask_ts_lh = np.zeros(len(effect_lh))
    # mask_ts_lh[sig_lh] = y_max
#
    # axs_t[0].plot(time_plot, mask_ts_lh, 'k')
    # axs_t[0].legend()
    #axs_t[0].plot(time_plot[mask_ts_lh !=0],mask_ts_lh[mask_ts_lh!=0],label='lh')
    # axs_t[1].plot(time_plot, np.mean(data_rh_list[0], axis=0),
    # label=f'{search_lab[0]}_rh')
    # axs_t[1].plot(time_plot, np.mean(data_rh_list[1], axis=0),
    # label=f'{search_lab[1]}_rh')
    # axs_t[1].plot(time_plot, np.mean(data_rh_list[2], axis=0),
    # label=f'{search_lab[2]}_rh')
    # axs_t[1].plot(time_plot, np.mean(data_rh_list[3], axis=0),
    # label=f'{search_lab[3]}_rh')
    # y_min, y_max = axs_t[1].get_ylim()
    # mask_ts_rh = np.zeros(len(effect_rh))
    # mask_ts_rh[sig_rh] = y_max
#
    # axs_t[1].plot(time_plot, mask_ts_rh, 'k')
    # axs_t[1].legend()
#
    # fig_t.suptitle(
    # f'Effect of {effect_label} for label: {stc_label[0].name[:-3]}')
    # fig_m, axs_m = plt.subplots(2)
    # axs_m[0].plot(time_plot, effect_lh, label='lh')
    #axs_t[0].plot(time_plot[mask_ts_lh !=0],mask_ts_lh[mask_ts_lh!=0],label='lh')
    # axs_m[1].plot(time_plot, effect_rh, label='rh')
    #axs_t[1].plot(time_plot[mask_ts_rh !=0],mask_ts_rh[mask_ts_rh!=0],label='rh')


def get_DF_clus(evoked, crop_value=None, g_excl=None):
    '''


    Parameters
    ----------
    evoked : TYPE
        DESCRIPTION. list containing 2 evokeds mne object
    crop_value : TYPE, optional
        DESCRIPTION. The default is None. touple cointaining time limits
    g_excl : TYPE, optional
        DESCRIPTION. The default is None. list containing subjects you want to exclude

    Returns
    -------
    X : TYPE
        DESCRIPTION. formatted subject data

    '''
    if crop_value is not None:
        data_crop = evoked[0].crop(crop_value[0], crop_value[1])
        data_shape = data_crop.data.shape
        subj_len = len(evoked)

    else:
        data_shape = evoked[0].data.shape
        subj_len = len(evoked)
    if g_excl!= None:
        subj_len=len(evoked)-len(g_excl)
        evoked=[ev for ev in evoked if not any(g in ev.comment for g in g_excl) ]

    X = np.empty((subj_len, data_shape[1], data_shape[0]))

    if crop_value is not None:
        for idx, ev in enumerate(evoked):
            X[idx, :, :] = ev.crop(crop_value[0], crop_value[1]).data.T

        print(np.shape(X))

    else:
        for idx, ev in enumerate(evoked):
            X[idx, :, :] = ev.data.T

    return X


##
def get_tTest_cluster(X, list_stc, src, label=None, FDR=False, p_val=.05,
                      report=None, crop_value=None, n_perm=50, thresh=None, time_smooth=False, t_pow=1):
    adj = mne.spatial_src_adjacency(src=src)
    n_suj = np.shape(X)[0]
    print(label)
    print(n_suj)
    clus_thresh = 0.1
    print(f'clus threshold for display is {clus_thresh}')
    if '_o_' in label:
        cond_lab='oral'
    elif '_n_' in label:
        cond_lab='nasal'
    else:
        raise ValueError('Cannot determine condition')

    if thresh == None:
        t_threshold = -stats.distributions.t.ppf(p_val / 2., n_suj - 1)
    elif thresh == 'TFCE':
        t_threshold = dict(start=0.5, step=.1)

    def t_test_func(*args):
        return ttest_1samp_no_p(args[0], sigma=0)

    if time_smooth:
        X = np.mean(X, 1)
        print(np.shape(X))
        T_obs, clusters, cluster_p_values, H0 = clu = \
            permutation_cluster_test(
                [X], adjacency=adj, n_jobs=-1, threshold=t_threshold,
                n_permutations=n_perm, t_power=t_pow, stat_fun=t_test_func)
    else:
        print(np.shape(X))
        T_obs, clusters, cluster_p_values, H0 = clu = \
            spatio_temporal_cluster_1samp_test(
                X, adjacency=adj, n_jobs=-1, threshold=t_threshold,
                n_permutations=n_perm, t_power=t_pow)
    if not any(cluster_p_values < clus_thresh):
        dir_stc = f'ana/MNE/cluster_stats/ttests/{cond_lab}/'
        log = [['Analyses info:'], [f'cluster pvalue: {p_val}'],
               [f'Time smooth is {time_smooth}'],
               [f'Time range is {int(crop_value[0]*1000)} to {int(crop_value[1]*1000)} ms,'],
               [f'Number of permutations: {n_perm}'],
               [f'T statistics threshold for clustering is: {t_threshold}']]

        filename = dir_stc + label
        print('writing log to file')
        if not os.path.exists(dir_stc):
            os.mkdir(dir_stc)
        with open(filename + '.txt', 'w') as file:
            file.write('No significant cluster for this analysis\n')
            for ob in log:
                file.write(ob[0] + '\n')
        return None, clu

    good_cluster_idx = np.where(cluster_p_values < clus_thresh)[0]
    good_cluster_value = cluster_p_values[good_cluster_idx]

    log = [['Analyses info:'], [f'cluster pvalue: {p_val}'],
           [f'Time smooth is {time_smooth}'],
           [f'Time range is {int(crop_value[0]*1000)} to {int(crop_value[1]*1000)} ms,'],
           [f'Number of permutations: {n_perm}'],
           [f'T statistics threshold for clustering is: {t_threshold}'],
           [f'index of good clusters {good_cluster_idx} and values: {good_cluster_value}']
           ]

    print("visualizing clusters")
    tstep = list_stc[0].tstep
    print(f'this is the Tstep: {tstep}')

    if time_smooth:
        if thresh == 'TFCE':
            T_obs_plot = np.zeros(T_obs.shape)
            mask_clus = np.where(cluster_p_values < clus_thresh)

            T_obs_plot[mask_clus] = T_obs[mask_clus]
            stc_all_cluster = stc_plot(
                T_obs_plot, list_stc, tmin=crop_value[0])
            stc_all_cluster.plot('fsaverage', hemi='split')
        else:

            for clus, pval in zip(clusters, cluster_p_values):

                if pval < clus_thresh:
                    T_obs_plot = np.zeros(T_obs.shape)
                    T_obs_plot[clus] = T_obs[clus]
                    stc_all_cluster = stc_plot(
                        T_obs_plot, list_stc, tmin=crop_value[0])
                    stc_all_cluster.plot('fsaverage', hemi='split')

    else:
        stc_all_cluster = summarize_clusters_stc(
            clu, p_thresh=clus_thresh, tstep=tstep, vertices=list_stc[0].vertices, subject='fsaverage')
        brain = stc_all_cluster.plot(hemi='split', subject='fsaverage',
                                     size=(600, 600))

    if label is not None and stc_all_cluster is not None:
        #captions = [f'Time-course of {label}']
        #report.add_stc(stc_all_cluster, title=captions[0], subject='fsaverage')

        dir_stc = f'ana/MNE/cluster_stats/ttests/{cond_lab}/{label}/'
        filename = dir_stc + label
        if not os.path.exists(dir_stc):
            os.mkdir(dir_stc)
        stc_all_cluster.save(filename,overwrite=True)
        with open(filename + '.txt', 'w') as file:
            for ob in log:
                file.write(ob[0] + '\n')
        for clu_idx in good_cluster_idx:
            filename_clu = filename + f'_clus_{clu_idx}'
            np.save(filename_clu, clusters[clu_idx])

        return stc_all_cluster, clu

    else:
        return stc_all_cluster, clu


def plot_stc_clu(clu, pval=0.05):
    t_obs, clusters, cluster_pv, H0 = clu

    n_sig_clu = len(np.where(cluster_pv < pval)[0])
    T_obs_plot = 0*np.ones_like(t_obs)
    for c, p_val in zip(clusters, cluster_pv):
        if p_val <= pval:
            print('found sig clu')

            for t_idx, v_idx in zip(c[0], c[1]):
                count = np.shape(np.where(c[1] == [v_idx]))[1]
                T_obs_plot[t_idx, v_idx] = t_obs[t_idx, v_idx]
    return T_obs_plot


## Cluster t-test wrapper
def tTest_cluster_ana(stc_files, search_lab, src, label=None, crop_value=None, FDR=False, p_val=.05, g_excl=None, thresh=None,
                      report=None, n_perm=50):
    stc_1 = get_list_cond(stc_files, search_lab[0])
    stc_2 = get_list_cond(stc_files, search_lab[1])

    X_1 = get_DF_clus(stc_1, crop_value=crop_value, g_excl=g_excl)
    X_2 = get_DF_clus(stc_2, crop_value=crop_value, g_excl=g_excl)

    # X_1_T=X_1.reshape(20484,181,len(stc_1))
    # X_2_T=X_2.reshape(20484,181,len(stc_1))
    X = X_1 - X_2

    # X_P=[X_1,X_2]
    # X_T=X_P.reshape()
    # X=X_T[:,:,:,0]-X_T[:,:,:,1]
    print(np.shape(X))

    n_rep = len(stc_1)
    len_check([stc_1, stc_2], n_rep)

    if label != None:
        report, stc_all_cluster, clu = get_tTest_cluster(X, stc_1, src=src, label=label, FDR=FDR, p_val=p_val,
                                                         report=report,
                                                         crop_value=crop_value, n_perm=n_perm, thresh=thresh)
        return report, stc_all_cluster, clu
    else:
        stc_all_cluster, clu = get_tTest_cluster(X, stc_1, src=src, label=label, FDR=FDR, p_val=p_val,
                                                 report=report,
                                                 crop_value=crop_value, n_perm=n_perm, thresh=thresh)
        T_obs_plot = plot_stc_clu(clu)

        stc_time_course = stc_plot(T_obs_plot.T, stc_1, tmin=crop_value[0])
        brain = stc_time_course.plot(hemi='both', subject='fsaverage',
                                     size=(600, 600))

        return stc_all_cluster, clu, stc_time_course


## Anovas
def Anova_core(evoked, crop_value=None, g_excl=None, factor_levels=[2, 2],
               effects='A*B'):
    # get raw X
    X = [get_DF(X, crop_value=crop_value, g_excl=g_excl) for X in evoked]
    print(np.shape(X))
    # format X
    ## define dimensions
    n_rep = len(evoked[0])
    n_conditions = len(evoked)
    n_chan = np.shape(X)[2]
    print(n_chan)
    n_times = np.shape(X)[3]
    print(n_times)
    ## check that all evoked have same number of subjects
    len_check(evoked, n_rep)
    ## reformat data
    data = np.swapaxes(np.asarray(X), 1, 0)
    print(np.shape(data))
    # reshape last two dimensions in one mass-univariate observation-vector
    data = data.reshape(n_rep, n_conditions, n_chan * n_times)

    print(data.shape)

    # Compute Anova
    fvals, pvals = f_mway_rm(
            data, factor_levels, effects=effects, correction=True)

    return fvals, pvals


def Anovas_ana(evoked, effects_labels, crop_value=None, g_excl=None, factor_levels=[2, 2],
               effects='A*B', FDR=False, report=None, p_val=0.05, time_smooth=False):

    # get raw X
    X = [get_DF(X, crop_value=crop_value, g_excl=g_excl) for X in evoked]
    print(np.shape(X))
    # format X
    ## define dimensions
    n_rep = len(evoked[0])
    n_conditions = len(evoked)
    n_chan = np.shape(X)[2]
    print(n_chan)
    n_times = np.shape(X)[3]
    print(n_times)
    ## check that all evoked have same number of subjects
    len_check(evoked, n_rep)
    ## reformat data
    data = np.swapaxes(np.asarray(X), 1, 0)
    print(np.shape(data))
    if time_smooth:
        data = np.mean(data, 3)
        print(data.shape)
        data = data.reshape(n_rep, n_conditions, n_chan)
    # reshape last two dimensions in one mass-univariate observation-vector
    else:
        data = data.reshape(n_rep, n_conditions, n_chan * n_times)

    print(data.shape)

    # Compute Anova
    fvals, pvals = f_mway_rm(
        data, factor_levels, effects=effects, correction=True)
    # Plot anova
    #fvals = [fvals]
    #pvals = [pvals]
    stc_obj_list = []

    for effect, sig, labels in zip(fvals, pvals, effects_labels):

        # evoked
        if not time_smooth:
            sig_mask = sig.reshape(n_chan, n_times)
            effect_fdr = effect.reshape(n_chan, n_times)
        else:
            effect_fdr = effect
            sig_mask = sig
        if FDR:
            reject, pval = fdr_correction(sig_mask)
            fdr_log = '_FDR_'
        else:
            reject = sig_mask < p_val
            fdr_log = '_noc_'

        effect_fdr[reject == False] = 0
        stc_obj = stc_plot(effect_fdr, evoked[0], tmin=crop_value[0])

        kwargs = dict(hemi='both', subject='fsaverage',
                      size=(600, 600))
        brain_auto = stc_obj.plot(figure=None, **kwargs)
        if time_smooth:
            dir_stc = f'ana/MNE/stats/inverse/anovas/{effects_labels[2]}{fdr_log}{p_val}_{crop_value[0]}-{crop_value[1]}/'
        else:
            dir_stc = f'ana/MNE/stats/inverse/anovas/{effects_labels[2]}{fdr_log}{p_val}/'
        filename = dir_stc + labels
        if not os.path.exists(dir_stc):
            os.mkdir(dir_stc)
        stc_obj.save(filename)
        stc_obj_list.append(stc_obj)

    return stc_obj_list, pvals


def Anovas_cluster(stc_list, src, crop_value=None, g_excl=None, factor_levels=[2, 2],
                   effects='A:B', n_perm=50, thresh=None, label=None, time_smooth=False, exclude_mask=None, pval=0.05):
    adj = mne.spatial_src_adjacency(src=src)
    clus_thresh = 0.1

    X_list = [get_DF_clus(stc, crop_value=crop_value, g_excl=g_excl)
              for stc in stc_list]
    print(np.shape(X_list))
    n_rep = len(stc_list[0])
    n_conditions = len(stc_list)
    n_chan = np.shape(X_list)[2]
    print(n_chan)
    n_times = np.shape(X_list)[3]
    print(n_times)

    if thresh == None:
        f_threshold = f_threshold_mway_rm(
            n_rep, factor_levels, effects, pvalue=pval)
    elif thresh == 'TFCE':
        f_threshold = dict(start=2, step=1)

    if time_smooth:
        X_list = np.mean(X_list, 2)
        print(np.shape(X_list))
    # print(np.shape(data))
    # data = data.reshape(n_rep, n_conditions, n_chan * n_times

    def stat_fun_anov(*args):
        return f_mway_rm(np.swapaxes(args, 1, 0),
                         factor_levels=factor_levels, effects=effects,
                         return_pvals=False)[0]

    # def stat_fun_anov(*args):
        # return f_mway_rm(np.swapaxes(args, 1, 0),
        # factor_levels=factor_levels, effects=effects,
        #return_pvals=False)[0]

    # Compute Anova
    F_obs, clusters, clust_p, H0 = clu =\
        permutation_cluster_test(
            X_list, adjacency=adj, threshold=f_threshold, stat_fun=stat_fun_anov, n_permutations=n_perm, n_jobs=-1, t_power=0, exclude=exclude_mask)

    # F_obs, clusters, clust_p, H0 = clu =\
    # spatio_temporal_cluster_test(
    # X_list, adjacency=adj, threshold=f_threshold, stat_fun=stat_fun_anov, n_permutations=n_perm, n_jobs=-1)

    if not any(clust_p < clus_thresh):
        return F_obs, clusters, clust_p, H0
    if time_smooth:
        for clus, pval in zip(clusters, clust_p):
            if pval < clus_thresh:
                F_obs_plot = np.zeros(F_obs.shape)
                F_obs_plot[clus] = F_obs[clus]
                stc_all_cluster = stc_plot(
                    F_obs_plot, stc_list[0], tmin=crop_value[0])
                stc_all_cluster.plot('fsaverage', hemi='split')
        return F_obs, clusters, clust_p, H0
    # plot all cluster together
    stc_all_cluster = summarize_clusters_stc(
        clu, tstep=stc_list[0][0].tstep, vertices=stc_list[0][0].vertices, subject='fsaverage')
    brain = stc_all_cluster.plot(hemi='both', subject='fsaverage',
                                 size=(600, 600))
    T_obs_plot = plot_stc_clu(clu)

    stc_time_course = stc_plot(T_obs_plot.T, stc_list[0], tmin=crop_value[0])
    brain = stc_time_course.plot(hemi='both', subject='fsaverage',
                                 size=(600, 600))
    if label is not None:

        dir_stc = 'ana/MNE/cluster_stats/'
        filename = dir_stc + label
        if not os.path.exists(dir_stc):
            os.mkdir(dir_stc)
        stc_all_cluster.save(filename+'_all_clus')
        stc_time_course.save(filename+'_time_course')

    return stc_all_cluster, clu, stc_time_course


def get_max_activation_coord(src, stc):
    stc.data

    # plot time course of cluster

    # plot time course of cluster

    # plot time course of cluster
    # plot time course of cluster

#%%
