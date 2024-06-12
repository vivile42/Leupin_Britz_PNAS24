#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 09:59:40 2021

@author: leupinv
"""
import os
import platform



platform.system()

# define starting datafolder

if platform.system()=='Darwin':
    #os.chdir('/Users/leupinv/switchdrive/BBC/WP1/data/EEG/tsk/')
    os.chdir('/Volumes/BBC/BBC/WP1/data/EEG/tsk/')
    dir_fold=os.getcwd()
    base_datafolder='/Volumes/Elements/'
elif platform.system()=='Windows':
    os.chdir('Z:/BBC/WP1/data/EEG/tsk/')
    dir_fold=os.getcwd()
   # os.chdir('E:/BBC/WP1/data/EEG/tsk/')
    base_datafolder='E:/'
elif platform.system()=='Linux':
    os.chdir('Z:/BBC/WP1/data/EEG/tsk')
    dir_fold=os.getcwd()


# datafolder to get epochs from
eeg_format='fif'
datafolder='preproc'
conditions=['n']
folder='preproc'

# datafolder to get evoked from
datafolder_evo='ana/MNE/evo_list'

n_sources='5124'

raw_dir=dir_fold+'/raw/g01/g01_n_tsk_ds_eeg-raw.fif'
fwd_dir=dir_fold+f'/coregistration/BBC_{n_sources}sol-fwd.fif'

## covariance settings
cov_method='auto'
cov_tmin=0.9
cov_tmax=-0.1

## inverse operator settings

#inv_loose=1.
inv_depth=3

##
sys_lab=['maskNEG','maskON','maskOFF']
sys_mask=['sys_mask==0 | cardiac_phase=="sys"','sys_mask==1',
                        'maskOFF']

## apply inverse settings
inv_method = "dSPM"
snr = 3.
lambda2 = 1. / snr ** 2

loose=True
cov_end=True

if loose:
    source_ori='loose'
else:
    source_ori='fixed'
if cov_end:
    cov_end_cond='cov_end'
else:
    cov_end_cond='cov_start'

datafolder_inv=f'ana/MNE/source/{n_sources}_source/{source_ori}/{cov_end_cond}/depth_{inv_depth}/inv'

diffi_list=['normal','easy']

accuracy_cond=['correct','mistake']


heart_cond=['cfa','nc']

id_vep=['aware','unaware','dia','sys','inh','exh', 'aware/dia','unaware/dia','aware/sys',
                         'unaware/sys','aware/inh','unaware/inh','aware/exh','unaware/exh',
                         'aware/sys/inh','aware/sys/exh','aware/dia/inh','aware/dia/exh',
                         'unaware/sys/inh','unaware/sys/exh','unaware/dia/inh','unaware/dia/exh',
                         'sys/inh','sys/exh','dia/inh','dia/exh']

id_hep_type=['R','R2','T','T2']

comb_type=['aware','unaware','inh','exh']



id_hep=['/'.join([x,y]) for x in id_hep_type for y in comb_type]

id_hep2=['/'.join([x,y,z]) for x in id_hep_type for y in comb_type[:2] for z in comb_type[-2:]]

id_hep3=['RRCA','RRCU']

id_hep_fin =id_hep_type+id_hep+id_hep2+id_hep3

print(id_hep)
id_hep=['aware','unaware','dia','sys','inh','exh', 'aware/dia','CU/dia','aware/sys',
                         'unaware/sys','aware/inh','unaware/inh','aware/exh','unaware/exh']


id_xns=['aware','unaware','dia','sys','inh','exh', 'aware/dia','unaware/dia','aware/sys',
                         'unaware/sys','aware/inh','unaware/inh','aware/exh','unaware/exh']
