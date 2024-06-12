# Get ERPs from cleaned epochs

from base.files_in_out import GetFiles, filter_list

import evoked.evoked_constants as cs
import base.base_constants as b_cs

from evoked.evoked_MNE_helper import EpochGroup
import gc
import mne

# In[Get list of epochs]
import os
import platform


platform.system()

# define starting datafolder

if platform.system() == 'Darwin':
    os.chdir('/Volumes/BBC/BBC/WP1/data/EEG/tsk')
    base_datafolder = '/Volumes/Elements/'
else:
    os.chdir('Z:/BBC/WP1/data/EEG/tsk/')
    
sys_lab = cs.sys_lab

for idx, sys in enumerate(sys_lab):
    for cfa in cs.heart_cond:
        epochs_group = EpochGroup()
        for g_n in b_cs.G_N:

            for cond in cs.condition[0]:
                files = GetFiles(filepath=cs.datafolder,
                                 condition=cond, g_num=g_n,
                                 eeg_format='clean_epo.fif')

            for file in files.condition_files:
                epochs_group.add_epoch(file, g_num=g_n, idx=idx)

        epochs_group.get_all_list()
        epochs_group.get_averages(cfa,miss=False)
        del epochs_group
        gc.collect()

