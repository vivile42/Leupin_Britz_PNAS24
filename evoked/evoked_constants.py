# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 15:24:53 2021

@author: Viviana
"""

import os
import platform


platform.system()

# define starting datafolder

if platform.system() == 'Darwin':
    os.chdir('/Volumes/BBC/BBC/WP1/data/EEG/tsk')
    base_datafolder = '/Volumes/Elements/'
else:
    os.chdir('Z:/BBC/WP1/data/EEG/tsk/')


eeg_format = 'fif'
eeg_exp = 'tsk'
datafolder = 'preproc'
condition = ['n']


#sys_mask=['sys_mask==1', 'noh']


#sys_lab=['sysEAR','sysLAT','maskNEG','maskON','inhEAR','inhLAT','exhEAR','exhLAT','maskOFF']

sys_lab = ['maskON', 'maskOFF']


heart_cond = ['cfa','nc']

diffi_list = ['easy','normal']

accuracy_cond = ['correct','mistake']

id_vep = ['aware', 'unaware', 'dia', 'sys', 'inh', 'exh', 'aware/dia', 'unaware/dia', 'aware/sys',
          'unaware/sys', 'aware/inh', 'unaware/inh', 'aware/exh', 'unaware/exh',
                         'aware/sys/inh', 'aware/sys/exh', 'aware/dia/inh', 'aware/dia/exh',
                         'unaware/sys/inh', 'unaware/sys/exh', 'unaware/dia/inh', 'unaware/dia/exh',
                         'sys/inh', 'sys/exh', 'dia/inh', 'dia/exh']

id_hep_type = ['R', 'R2', 'T', 'T2']

comb_type = ['aware', 'unaware', 'inh', 'exh']


id_hep = ['/'.join([x, y]) for x in id_hep_type for y in comb_type]

id_hep2 = ['/'.join([x, y, z]) for x in id_hep_type for y in comb_type[:2]
           for z in comb_type[-2:]]

id_hep3 = ['RRCA', 'RRCU']

id_hep_fin = id_hep_type+id_hep+id_hep2+id_hep3

print(id_hep)
id_hep = ['aware', 'unaware', 'dia', 'sys', 'inh', 'exh', 'aware/dia', 'CU/dia', 'aware/sys',
          'unaware/sys', 'aware/inh', 'unaware/inh', 'aware/exh', 'unaware/exh']


id_xns = ['aware', 'unaware', 'dia', 'sys', 'inh', 'exh', 'aware/dia', 'unaware/dia', 'aware/sys',
          'unaware/sys', 'aware/inh', 'unaware/inh', 'aware/exh', 'unaware/exh']


# Constants convert to eeglab
eeg_format_conv = 'vep_clean_epo.fif'


