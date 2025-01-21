# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 09:59:23 2021

@author: Engi
"""
import pandas as pd
import numpy as np


class Behav_DF():
    def __init__(self, files):
        self.files = files
        self.df = pd.read_feather(files.filt[0])
        self.df['g_num'] = self.files.g_num
        self.df['condition'] = self.files.condition
        self.df_stim = self.df[self.df['signal_type'] == 'vep']
        self.df_stim = self.df_stim[self.df_stim['difficulty'] == 'normal']



    def find_thresh(self):
        val = self.df_stim.mrk_awa.value_counts()
        tot = np.sum(val)
        #uses mrk, sums CA CU which are the first 2
        tot_corr = np.sum(val[:2])

        CACU_corr = np.round(val['CA']/tot_corr, 2)
        corr = np.round(tot_corr/tot, 2)
        self.df['CACU_corr'] = CACU_corr
        self.df['corr'] = corr

    def compute_distribution(self):
        val = self.df_stim.mrk_card_awa.value_counts()
        CACU_sys = val['CAS']/(val['CAS']+val['CUS'])
        CACU_dia = val['CAD']/(val['CAD']+val['CUD'])
        self.df['CACU_sys'] = CACU_sys
        self.df['CACU_dia'] = CACU_dia


def modified_z_scores(df, correction=1.4826):
    arr_df = df['RT'].to_numpy()
    median = np.median(arr_df)
    deviation_from_med = arr_df-median
    mad = np.median(np.abs(deviation_from_med))
    mod_zscores = deviation_from_med/(correction*mad)
    return mod_zscores, mad




def cut_list(list_cut, df, q=3):
    for x in list_cut:
        df[f'{x}_bin'] = pd.qcut(df[x], q=q)

    return df
