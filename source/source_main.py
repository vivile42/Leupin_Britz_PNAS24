#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 14:51:29 2021

@author: leupinv
"""
import base.base_constants as b_cs
from base.files_in_out import GetFiles,filter_list,getListOfFiles
import source.source_constants as cs
import source.source_helper as source_hp
import mne
import os



# raw=mne.io.read_raw_fif(cs.raw_dir)
# info_raw=raw.info


# for g_n in b_cs.G_N:
#     for cond in cs.conditions:
#         files = GetFiles(filepath=cs.datafolder,
#                            condition=cond,g_num=g_n,
#                            eeg_format='clean_epo.fif')
#         for epo_file in files.condition_files:
#             if 'xns' not in epo_file and 'hep' not in epo_file:
#                 cov=source_hp.GetCovariance(epo_file, g_n,info_raw,loose=cs.loose,cov_end=cs.cov_end)
#                 cov.get_cov()
#                 cov.get_inverse()



#
for g_n in b_cs.G_N:
        for cond in cs.conditions:
            files_evo = GetFiles(filepath=cs.datafolder_evo,
                              condition=cond,g_num=None,
                              eeg_format='ave.fif')
            files_evo.condition_files=[x for x in files_evo.fflist if x.endswith('list-ave.fif')]
            files_inv = GetFiles(filepath=cs.datafolder_inv,
                              condition=cond,g_num=None,
                              eeg_format='inv.fif')
            files_inv.condition_files=[x for x in files_inv.fflist if x.endswith('-inv.fif')]
            #
            #
            for inv in files_inv.condition_files:
                inv_obj=mne.minimum_norm.read_inverse_operator(inv,verbose=False)
                if g_n in inv:
    #
                    for evo in files_evo.condition_files:
    #
                        inv_STC=source_hp.GetSTC(inv,inv_obj,evo, g_n,loose=cs.loose,cov_end=cs.cov_end,method=cs.inv_method)
    #
