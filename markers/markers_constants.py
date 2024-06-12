#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 09:36:30 2021

Constants used to run the markers module
@author: leupinv
"""
import os
import platform



platform.system()

# define starting datafolder

if platform.system()=='Darwin':
    os.chdir('/Volumes/BBC/BBC/WP1/data/EEG/tsk/')
    base_datafolder='/Volumes/Elements/'
else:
    #os.chdir('c:/Users/Engi/all/BBC/WP1/data/EEG/tsk')
    os.chdir('Z:/BBC/WP1/data/EEG/tsk/')
    base_datafolder='Z:/BBC/WP1/data/EEG/raw_nods'


eeg_format='bdf'
eeg_exp='tsk'
condition=['n']

#DF constants
method_ans=['pad','backfill']

## Outputs variables
#folders
type_sig_mrk_DF='mrk_DF'
type_sig_png='png'
type_sig_physig='phy_sig'
# file end
file_end_png='.png'
file_end_feather='.feather'
file_end_csv='.csv'

# DS+ filters parametres
#low end
l_filt=0.5


# high end
h_filt=40

# resample fr

sfreq=256

#♠ Get bad interval parametres
# interval minimum to estabish a pause
int_min=8
# buffer to be safe
buff_int=3

## get bad rsp parametres
# this can be manually adjusted for outlier subjects
#multiplier for the higher range of the std

up_std=2.5
#multiplier for the lower range of the std
low_std=1.5

