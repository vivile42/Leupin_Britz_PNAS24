#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 10:42:09 2021

@author: leupinv
"""
import os
import platform



platform.system()

# define starting datafolder 

if platform.system()=='Darwin':
    os.chdir('/Users/leupinv/switchdrive/BBC/WP1/data/EEG/tsk/')
    base_datafolder='/Volumes/Elements/'
else:
    os.chdir('Z:/BBC/WP1/data/EEG/tsk')
    #os.chdir('c:/Users/Engi/all/BBC/WP1/data/EEG/tsk')
    base_datafolder='E:/'

eeg_format='fif'
eeg_exp='tsk'
datafolder='raw'
condition=['n']


## Outputs variables
#folders
type_sig_mrk_DF='mrk_DF'
type_sig_png='png'
type_sig_physig='phy_sig'
# file end
file_end_png='.png'
file_end_feather='.feather'
file_end_csv='.csv'


# tmin max
tmin, tmax = -0.2, 1

baseline = (None) #no baseline correction