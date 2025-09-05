#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 09:58:02 2021

@author: leupinv
"""
## libraries
import numpy as np
import mne
from scipy import stats
from mne.stats import fdr_correction, f_mway_rm, permutation_t_test
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['svg.fonttype']='none'
mpl.rcParams['text.usetex']=False
import os
import pingouin as pg
import multiprocessing as mp
import seaborn as sns

## General use functions

def define_period(region,phy_sig=None):
    if region=="P1":
        if phy_sig=='rsp_phase':
            crop_period=[0.065,0.095]

        elif phy_sig=='cardiac_phase':
            crop_period=[0.09,0.12]
        else:
            raise ValueError('physiological signal, either rsp_phase or cardiac phase, '
                             'are required')
    elif region=="VAN":
        crop_period=[0.24,0.34]
    elif region=="LP":
        crop_period=[0.4,0.5]
    else:
        raise ValueError('Region not available, please choose from P1, VAN or LP')

    return crop_period

def get_erp_df(evokeds,crop_period,picks_ERP):
    erp_df={}
    for lab,cond in evokeds.items():
        evoked_1=get_DF(evokeds[lab].copy(), crop_value=crop_period,
                             picks=picks_ERP)
        mean_el=np.mean(evoked_1,1)
        mean_erp=np.mean(mean_el,1)
        mean_erp=mean_erp.tolist()
        erp_df[lab]=mean_erp

    return erp_df

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
    filter_list = [x for x in list_ if value in x]
    return filter_list[0]


def filter_list_equal(list_: list, value: str) -> list:
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
    filter_test = [x.replace("\\","/") for x in list_]
    print(filter_test[-1])
    filter_list = [x for x in filter_test if value == x.split("/")[-1][:-8]]

    return filter_list[0]

## T-test functions


def get_DF(evoked, crop_value=None, g_excl=None, picks=None):
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

    if picks is not None:
        evoked = [ev.pick(picks) for ev in evoked]
    if crop_value != None:
        data_crop = evoked[0].copy().crop(crop_value[0], crop_value[1])
        data_shape = data_crop.data.shape
        subj_len = len(evoked)

    else:
        data_shape = evoked[0].data.shape
        subj_len = len(evoked)
    if g_excl != None:
        subj_len = len(evoked)-len(g_excl)
        evoked = [ev for ev in evoked if not any(
            g in ev.comment for g in g_excl)]
        print(evoked)

    X = np.empty((subj_len, data_shape[0], data_shape[1]))

    if crop_value is not None:
        for idx, ev in enumerate(evoked):
            X[idx, :, :] = ev.copy().crop(crop_value[0], crop_value[1]).data

        print(np.shape(X))

    else:
        for idx, ev in enumerate(evoked):
            X[idx, :, :] = ev.data

    return X


def len_check(evoked, length):
    truth_seeker = [len(x) == length for x in evoked]
    if all(truth_seeker) == False:

        raise Exception('Unequal number of subjects')


def get_y_ps(ts, ps, sig, pos=True):
    if pos:
        ps = ps[ts > 0]
        ts = ts[ts > 0]

    else:
        ps = ps[ts < 0]
        ts = ts[ts < 0]

    sig_value_01 = ps.copy()
    sig_value_01[sig_value_01 > sig] = np.NaN
    y_01_idx = np.where(sig_value_01 == np.nanmax(sig_value_01))
    y_01 = ts[y_01_idx]
    return y_01


def get_1d_time(data_1d):
    mean_1d = np.mean(data_1d)
    sd_1d = np.std(data_1d, 0)
    return mean_1d, sd_1d


def get_resample(i, mean_X_1d):
    values = np.random.choice(mean_X_1d, size=len(mean_X_1d), replace=True)
    print(np.shape(values))
    mean_1d, sd_1d = get_1d_time(values)
    effect_size_1d = mean_1d/sd_1d
    return i, effect_size_1d


def get_bootstrap(mean_X_1d, n_rep=5000):
    pool = mp.Pool(mp.cpu_count())
    print(f'computing {n_rep} repetitions on {mp.cpu_count()} cores')

    result_objects = [pool.apply_async(
        get_resample, args=[i, mean_X_1d])for i in np.arange(n_rep)]
    boot = [r.get()[1] for r in result_objects]
    pool.close()
    pool.join()
    return boot


def get_tTest_picks(X, picks, FDR=False, plot_times='peaks', p_val=0.05,
                    crop_value=None, png=None, plot_sig_lines=True, color='b', effect_size=False,axes=False):
    X_data = X[0]-X[1]
    mean_X = np.mean(X_data, 1)
    mean_X_1d = np.mean(mean_X, 1)
    mean_D = np.mean(mean_X, 0)
    sd_D = np.std(mean_X, 0)
    out = stats.ttest_1samp(mean_X, 0, axis=0)
    ts = out[0]
    ps = out[1]

    if effect_size == 'whole':
        ts = mean_D/sd_D
        print(np.shape(mean_D))
        print(np.shape(sd_D))
        print("showing effect sizes")

    elif effect_size == '1d':
        # get 1D mean across time

        mean_1d, sd_1d = get_1d_time(mean_X_1d)

        effect_size_1d = mean_1d/sd_1d
        print(
            f'effect size for {int(crop_value[0]*1000)} to {int(crop_value[1]*1000)} time interval: {effect_size_1d}')
        boot = get_bootstrap(mean_X_1d)
        sns.distplot(boot)
        confidence = 0.95
        conf_int = np.percentile(
            boot, [100*(1-confidence)/2, 100*(1-(1-confidence)/2)])

        return effect_size_1d, conf_int

    mask_ts = ts.copy()
    if FDR:
        reject_fdr, pval_fdr = fdr_correction(ps)
        sig_value = reject_fdr
    else:
        sig_value = ps < p_val

    mask_ts[sig_value == False] = np.NaN
    y05_pos = get_y_ps(ts.copy(), ps.copy(), sig=0.05, pos=True)
    y05_neg = get_y_ps(ts.copy(), ps.copy(), sig=0.05, pos=False)
    y01_pos = get_y_ps(ts.copy(), ps.copy(), sig=0.01, pos=True)
    y01_neg = get_y_ps(ts.copy(), ps.copy(), sig=0.01, pos=False)
    y001_pos = get_y_ps(ts.copy(), ps.copy(), sig=0.001, pos=True)
    y001_neg = get_y_ps(ts.copy(), ps.copy(), sig=0.001, pos=False)
    y0001_pos = get_y_ps(ts.copy(), ps.copy(), sig=0.0001, pos=True)
    y0001_neg = get_y_ps(ts.copy(), ps.copy(), sig=0.0001, pos=False)

    print(np.shape(mean_X)[1])
    time = np.linspace(crop_value[0]*1000, crop_value[1]*1000, np.shape(mean_X)[1])
    if axes==False:
        fig,axes=plt.subplots()
    axes.plot(time, ts, label=png, color=color,linewidth=1.5)
    #plt.plot(time, mask_ts, 'k')

    axes.hlines(y=0, xmin=crop_value[0],
               xmax=crop_value[1], linewidth=0.8, colors='k', linestyle='-')

    if plot_sig_lines:
        xmax = -50
        axes.hlines(y=y05_pos, xmin=crop_value[0]*1000,
                   xmax=xmax, linewidth=1, colors='b', linestyle='-.')
        axes.hlines(y=y05_neg, xmin=crop_value[0]*1000,
                   xmax=xmax, linewidth=1, colors='b', linestyle='-.')

        axes.hlines(y=y01_pos, xmin=crop_value[0]*1000,
                   xmax=xmax, linewidth=1, colors='g', linestyle='-.')
        axes.hlines(y=y01_neg, xmin=crop_value[0]*1000,
                   xmax=xmax, linewidth=1, colors='g', linestyle='-.')
        axes.hlines(y=y001_pos, xmin=crop_value[0]*1000,
                   xmax=xmax, linewidth=1, colors='r', linestyle='-.')
        axes.hlines(y=y001_neg, xmin=crop_value[0]*1000,
                   xmax=xmax, linewidth=1, colors='r', linestyle='-.')
        axes.hlines(y=y0001_pos, xmin=crop_value[0]*1000,
                   xmax=xmax, linewidth=1, colors='m', linestyle='-.')
        axes.hlines(y=y0001_neg, xmin=crop_value[0]*1000,
                   xmax=xmax, linewidth=1, colors='m', linestyle='-.')
    axes.axvline(0, linewidth=0.9, color='k', linestyle='--')
    #axes.set_xticks(fontsize=10)
    if effect_size=='whole':
        axes.set_ylabel("VEP Cohen's D")
    else:
        axes.set_ylabel('T Values')
    axes.set_xlabel('Time (s)', fontsize=10)
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.set_xlim(crop_value[0]*1000-2,crop_value[1]*1000)


    axes.legend(prop={'size': 10})

    return ts


def get_tTest(X, label=None, FDR=False, plot_times='peaks',
              averages=None, p_val=.05, report=None, crop_value=None, g_excl=None, png=None,topo_limits=[-7.5,7.5]):
    '''



    Parameters
    ----------
    X : TYPE
        DESCRIPTION. Array containing formatted subject data
    label : TYPE, optional
        DESCRIPTION. The default is None. label to generate report with
    FDR : TYPE, optional
        DESCRIPTION. The default is False so NOC. Weather correct or not witf
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
    None.

    '''
    # T-test
    out = stats.ttest_1samp(X, 0, axis=0)
    ts = out[0]
    ps = out[1]

    # establish significancy mask (can be FDR or noc)
    if FDR:
        reject_fdr, pval_fdr = fdr_correction(ps)
        sig_value = reject_fdr
    else:
        sig_value = ps < p_val

    # plot image (need to convert in evoked onject to do it)
    print(crop_value[0])
    evok = evoked_plot(ts, tmin=crop_value[0])
    #generate whole picture
    fig_evo = evok.plot_image(
        mask=sig_value, scalings=1, units='T-value', show_names='auto',
        clim=dict(eeg=[-6, 6])
        )
    ax=plt.gca()
    y_ticks_loc=np.linspace(16,112,4)
    y_ticks_lab_loc=['post','right','ant','left']
    ax.set_yticks(y_ticks_loc)
    ax.set_yticklabels(y_ticks_lab_loc)
    if png != None:
        fig_path = f'ana/results_report/images/t-tests/{png}'
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
        filenam = fig_path+f'/{png}_mass.svg'
        fig_evo.savefig(filenam, dpi=1200,format='svg')
    #generate topoplot
    if plot_times != 'peaks':
        mask_topo = sig_value.copy()
        time = np.linspace(crop_value[0], crop_value[1], sig_value.shape[1])
        for plot_t in plot_times:
            time_idx = np.where(np.logical_and(
                time > plot_t-averages, time < plot_t+averages))[0]
            int_sig = mask_topo[:, time_idx]
            for n, line in enumerate(int_sig):
                if sum(line) >= 1:
                    sig_value[n, time_idx] = True
                    print(sum(line))
                    print(len(line))
                else:
                    sig_value[n, time_idx] = False

        print(sig_value.shape)

    fig_topo = evok.plot_topomap(
        plot_times, outlines='head', scalings=1, units='T-value', average=averages, mask=sig_value, vmin=topo_limits[0],
        vmax=topo_limits[1])
    if png != None:
        fig = plt.gcf()
        filenam = fig_path+f'/{png}_topo.svg'
        fig.savefig(filenam, dpi=1200, transparent=True,format='svg')
        filename_t=fig_path+f'/t_values_{png}.ep'

        np.savetxt(filename_t,ts.T)
        filename_p=fig_path+f'/p_values_noc_{png}.ep'
        np.savetxt(filename_p,ps.T)
        if FDR:
            filename_p=fig_path+f'/p_values_FDR_{png}.ep'
            np.savetxt(filename_p,pval_fdr.T)

    # add graphs to report to produce HTML only if label is present
    if FDR:
        corr = 'FDR corrected'
    else:
        corr = 'noc'

    if label != None:
        captions = [f'Time-course of {label} {corr}, pval ={p_val}',
                    f'Topoplot of {label} on {plot_times} {corr}, p val={p_val}']
        report.add_figure(
            fig_evo, title=f'T-tests for {label}: image', caption=captions[0], image_format='svg')
        report.add_figure(
            fig_topo, title=f'T-tests for {label}: topo', caption=captions[1], image_format='svg')
        return report
    else:
        return ts, ps


def get_perm_tTest(X, n_perm=1000, label=None, FDR=False, plot_times='peaks', averages=None, p_val=.05, report=None, crop_value=None, g_excl=None, png=None):
    # format data
    shape = X.shape
    X = X.reshape(shape[0], shape[1]*shape[2])

    # T-test
    out = permutation_t_test(X, n_permutations=n_perm)
    ts = out[0]
    ps = out[1]
    H0 = out[2]
    ts = ts.reshape(shape[1], shape[2])
    ps = ps.reshape(shape[1], shape[2])
    # establish significancy mask (can be FDR or noc)
    if FDR:
        reject_fdr, pval_fdr = fdr_correction(ps)
        sig_value = reject_fdr
    else:
        sig_value = ps < p_val

    # plot image (need to convert in evoked onject to do it)
    print(shape)

    #ts=ts.reshape(shape[0],shape[1],shape,[2])
    #ps=ps.reshape(shape[0],shape[1],shape,[2])
    evok = evoked_plot(ts, tmin=crop_value[0])
    #generate whole picture
    fig_evo = evok.plot_image(
        mask=sig_value, scalings=1, units='T-value', show_names='auto')
    if png != None:
        fig_path = f'ana/results_report/images/t-tests/{png}'
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
        filenam = fig_path+f'/{png}_mass.png'
        #fig_evo.savefig(filenam, dpi=600)
    #generate topoplot

    fig_topo = evok.plot_topomap(
        plot_times, outlines='head', scalings=1, units='T-value', average=averages, mask=sig_value)
    if png != None:
        fig = plt.gcf()
        filenam = fig_path+f'/{png}_topo.png'
        #fig.savefig(filenam, dpi=600, transparent=True)
        filename_t=fig_path+f'/t_values_{png}.ep'

        np.savetxt(filename_t,ts.T)
        filename_p=fig_path+f'/p_values_noc_{png}.ep'
        np.savetxt(filename_p,ps.T)
        if FDR:
            filename_p=fig_path+f'/p_values_FDR_{png}.ep'
            np.savetxt(filename_p,pval_fdr.T)


    # add graphs to report to produce HTML only if label is present
    if FDR:
        corr = 'FDR corrected'
    else:
        corr = 'noc'

    if label != None:
        captions = [f'Time-course of {label} {corr}, pval ={p_val}',
                    f'Topoplot of {label} on {plot_times} {corr}, p val={p_val}']
        report.add_figure(
            fig_evo, title=f'T-tests for {label}: image', caption=captions[0], image_format='svg')
        report.add_figure(
            fig_topo, title=f'T-tests for {label}: topo', caption=captions[1], image_format='svg')
        return report, H0
    else:
        return ts, ps, H0


def evoked_plot(evok_effect, tmin):
    biosemi_montage = mne.channels.make_standard_montage('biosemi128')

    info = mne.create_info(ch_names=biosemi_montage.ch_names, sfreq=256.,
                           ch_types='eeg')
    evok = mne.EvokedArray(evok_effect, info, tmin=tmin)
    evok.set_montage(biosemi_montage)
    return evok


def tTest_ana(evoked, label=None, crop_value=None, FDR=False,
              plot_times='peaks', averages=None, p_val=.05,
              g_excl=None, report=None, png=None, perm=False, n_perm=1000,
              picks=None, plot_sig_lines=True, color='b', effect_size=False,axes=False,topo_limits=[-7.5,7.5]):
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
    evoked_1 = get_DF(evoked[0], crop_value=crop_value,
                      g_excl=g_excl, picks=picks)
    evoked_2 = get_DF(evoked[1], crop_value=crop_value,
                      g_excl=g_excl, picks=picks)

    n_rep = len(evoked[0])
    len_check(evoked, n_rep)

    if picks is not None:
        X = [evoked_1, evoked_2]
        ts = get_tTest_picks(X, picks=picks, FDR=FDR, plot_times=plot_times,
                             p_val=p_val, crop_value=crop_value, png=png,
                             plot_sig_lines=plot_sig_lines, color=color,
                             effect_size=effect_size,axes=axes)
        return ts
    X = evoked_1-evoked_2
    if perm:
        H0 = get_perm_tTest(X, n_perm=n_perm, label=label, FDR=FDR, plot_times=plot_times,
                            averages=averages, p_val=p_val, report=report,
                            crop_value=crop_value, g_excl=g_excl, png=png)

    else:
        ts = get_tTest(X, label=label, FDR=FDR, plot_times=plot_times,
                       averages=averages,
                       p_val=p_val, report=report, crop_value=crop_value,
                       g_excl=g_excl, png=png,topo_limits=topo_limits)

    if label != None:
        return report
    elif perm:
        return H0
    elif label == None:
        return ts

## Bayes tests


def tTest_bayes(evoked, label=None, crop_value=None, FDR=False,
                plot_times='peaks', averages=None, p_val=.05, g_excl=None,
                report=None, png=None, perm=False, n_perm=1000):

    evoked_1 = get_DF(evoked[0], crop_value=crop_value, g_excl=g_excl)
    evoked_2 = get_DF(evoked[1], crop_value=crop_value, g_excl=g_excl)

    n_rep = len(evoked[0])
    len_check(evoked, n_rep)

    X = evoked_1-evoked_2
    print(X)
    # format data
    shape = X.shape
    X = X.reshape(shape[0], shape[1]*shape[2])

    # T-test
    out = pg.ttest(X, 0)
    return out
    # ts=out[0]
    # ps=out[1]
    # ts=ts.reshape(128, 257)
    # ps=ps.reshape(128,257)
    ##establish significancy mask (can be FDR or noc)
    # if FDR:
    # reject_fdr, pval_fdr = fdr_correction(ps)
    # sig_value=reject_fdr
    # else:
    # sig_value=ps<p_val
#
    #plot image (need to convert in evoked onject to do it)
    # print(shape)
#
#
    #ts=ts.reshape(shape[0],shape[1],shape,[2])
    #ps=ps.reshape(shape[0],shape[1],shape,[2])
    # evok=evoked_plot(ts,tmin=crop_value[0])
    #generate whole picture
    # fig_evo=evok.plot_image(mask=sig_value,scalings=1,units='T-value',show_names='auto')
    # if png != None:
    # fig_path=f'ana/results_report/images/t-tests/{png}'
    # if not os.path.exists(fig_path):
    # os.makedirs(fig_path)
    # filenam=fig_path+f'/{png}_mass.png'
    # fig_evo.savefig(filenam,dpi=600)
    #generate topoplot
#
    # fig_topo=evok.plot_topomap(plot_times,outlines='head',scalings=1,units='T-value',average=averages,mask=sig_value)
    # if png != None:
    # fig=plt.gcf()
    # filenam=fig_path+f'/{png}_topo.png'
    # fig.savefig(filenam,dpi=600,transparent=True)
#
#
    #add graphs to report to produce HTML only if label is present
    # if FDR:
    # corr='FDR corrected'
    # else:
    # corr='noc'
#
    # if label != None:
    # captions=[f'Time-course of {label} {corr}, pval ={p_val}',f'Topoplot of {label} on {plot_times} {corr}, p val={p_val}']
    # report.add_figure(fig_evo,title=f'T-tests for {label}: image' ,caption=captions[0],image_format='svg')
    # report.add_figure(fig_topo,title=f'T-tests for {label}: topo' ,caption=captions[1],image_format='svg')
    # return report
#
    #
    # if label != None:
    # return report


## Anovas

def Anovas_picks(evoked, effect_labels, picks, crop_value=None,
                 factor_levels=[2, 2], effects='A*B', FDR=False, p_val=0.05, png=None):
    X = [get_DF(X, crop_value=crop_value, picks=picks) for X in evoked]
    n_rep = len(evoked[0])
    n_conditions = len(evoked)
    n_chan = len(picks)
    n_times = evoked[0][0].data.shape[1]
    data = np.swapaxes(np.asarray(X), 1, 0)
    print(np.shape(data))
    data = np.mean(data, 2)
    print(np.shape(data))
    time_plot = np.linspace(crop_value[0], crop_value[1], n_times)
    # Compute Anova
    fvals, pvals = f_mway_rm(
        data, factor_levels, effects=effects, correction=True)

    #Plot anova

    for effect, sig, effect_label in zip(fvals, pvals, effect_labels):

        # evoked
        mask_effect = effect.copy()

        if FDR:
            reject, pval = fdr_correction(sig)
        else:
            reject = sig < p_val
        mask_effect[reject == False] = np.NaN
        plt.figure()
        plt.plot(time_plot, effect)
        plt.plot(time_plot, mask_effect, 'k')


def Anovas_ana(evoked, effects_labels, crop_value=None, g_excl=None, factor_levels=[2, 2],
               effects='A*B', FDR=False, report=None, topo_times='peaks', p_val=0.05, plot_average=0.1, png=None):

    #get raw X
    X = [get_DF(X, crop_value=crop_value, g_excl=g_excl) for X in evoked]
    #format X
    ## define dimensions
    n_rep = len(evoked[0])
    n_conditions = len(evoked)
    n_chan = 128
    n_times = np.shape(X[0])[2]
    ## check that all evoked have same number of subjects
    len_check(evoked, n_rep)
    ## reformat data
    data = np.swapaxes(np.asarray(X), 1, 0)
    # reshape last two dimensions in one mass-univariate observation-vector
    data = data.reshape(n_rep, n_conditions, n_chan * n_times)

    print(data.shape)

    # Compute Anova
    fvals, pvals = f_mway_rm(
        data, factor_levels, effects=effects, correction=True)

    #Plot anova

    for effect, sig, effect_label in zip(fvals, pvals, effects_labels):

        # evoked
        data_fdr = sig.reshape(n_chan, n_times)
        if FDR:
            reject, pval = fdr_correction(data_fdr)
        else:
            reject = sig.reshape(n_chan, n_times) < p_val
        effect_fdr = effect.reshape(n_chan, n_times)
        evok = evoked_plot(effect_fdr, tmin=crop_value[0])
        if FDR:
            corr = 'FDR corrected'
        else:
            corr = 'noc'

        fig_effect = evok.plot_image(mask=reject, scalings=1, units='F-value',
                                     show_names='auto', clim=dict(eeg=[0, None]), cmap='turbo')
        if png != None:
            fig_path = f'ana/results_report/images/anovas/{png}'
            if not os.path.exists(fig_path):
                os.makedirs(fig_path)
            file = f'/{png}_{effect_label}_mass.svg'
            filenam = fig_path+file
            txt_filename = fig_path+f'/{png}_info.txt'
            caption1 = f'{file} : Time-course of {effect_label} {corr}, pval ={p_val}\n'
            with open(txt_filename, 'a') as file:
                file.write(caption1)
            fig_effect.savefig(filenam, dpi=1200,format='svg')

        if topo_times != 'peaks':
            mask_topo = reject.copy()
            time = np.linspace(crop_value[0], crop_value[1], reject.shape[1])
            for plot_t in topo_times:
                time_idx = np.where(np.logical_and(
                    time > plot_t-plot_average, time < plot_t+plot_average))[0]
                int_sig = mask_topo[:, time_idx]
                for n, line in enumerate(int_sig):
                    if sum(line) >= 3:
                        print(sum(line))
                        print(len(line))
                        reject[n, time_idx] = True
                    else:
                        reject[n, time_idx] = False

            print(reject.shape)
        fig_topo = evok.plot_topomap(topo_times, outlines='head', scalings=1,
                                     vmin=0, cmap='turbo', units='F-value', average=plot_average, mask=reject)
        if png != None:
            fig_path = f'ana/results_report/images/anovas/{png}'
            if not os.path.exists(fig_path):
                os.makedirs(fig_path)
            file = f'/{png}_{effect_label}_topo.png'
            filenam = fig_path+file
            txt_filename = fig_path+f'/{png}_info.txt'
            caption2 = f'{file} : Topoplot of {effect_label} on {topo_times} {corr}, p val={p_val}\n'
            with open(txt_filename, 'a') as file:
                file.write(caption2)
            fig_topo.savefig(filenam, dpi=600, transparent=True)

        if report != None:

            # add graphs to report to produce HTML only if label is present

            caption1 = f'Time-course of {effect_label} {corr}, pval ={p_val}'
            caption2 = f'Topoplot of {effect_label} on {topo_times} {corr}, p val={p_val}'

            report.add_figure(
                fig=fig_effect, title=f'Anova {effects_labels[0]} X {effects_labels[1]}:  {effect_label} image ', caption=caption1, image_format='svg')
            report.add_figure(
                fig=fig_topo, title=f'Anova {effects_labels[0]} X {effects_labels[1]}:  {effect_label} topoplot ', caption=caption2, image_format='svg')

    return report
    return report
    return report
    return report
    return report
    return report
