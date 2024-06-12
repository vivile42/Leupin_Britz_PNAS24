#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 09:35:43 2021

@author: leupinv
"""
import markers.markers_constants as cs
import markers.markers_helper as hp

import base.files_in_out as files_in_out
import base.base_constants as b_cs
import markers.markers_MNE_helper as mne_hp
import matplotlib.pyplot as plt
import gc


for g_n in b_cs.G_N:
    for cond in cs.condition[0]:
        files = files_in_out.GetFiles(
            filepath=cs.base_datafolder, condition=cond, g_num=g_n)
        tskfiles = files.condition_files
        rsp_sig_list = []
        card_sig_list = []
        raw_list = []

        for idx in range(files.condition_nfiles):

            files.get_info(idx)
            mne_data = mne_hp.MarkersMNE(files)
            raw_list.append(mne_data.raw)

        mne_data.merge_raws(raw_list) #merge EEGs dataframes
        mne_data.get_ds_eeg(mne_data.raws, open_file=False) #downsample EEG
        mne_data.get_triggers()
        mne_data.get_card(raws=mne_data.raws) #process cardiac signal
        mne_data.get_rsp(raws=mne_data.raws) #process respiratory signal
        mne_data.merge_rsp_DF() # get respiratory dataframe for triggers
        mne_data.get_ecg_stim_DF() #get cardiac dataframe for triggers
        mne_data.get_ecg_hep_DF() # get dataframe to extract heartbeaat evoked potentials (extra)

        DF_class=hp.DF_Markers(mne_data) #get markers to generate annotations for triggers
        annot=DF_class.get_annotations()
        DF_class.get_metadata() #generate metadata

        mne_data.update_annot(annot=annot,append=True)
        mne_data.get_HRV() #compute heart rate varaibility
        rsa,rrv=mne_data.get_RSA() #compute respiratory sinus arrhythmia
        mne_data.save_df()
        files_in_out.save_report(files,mne_data.report,short=True,final=False)
        files_in_out.save_report(files,mne_data.report,short=True,final=True)
        #clean up ram
        plt.close('all') #close plots
        # delete huge variables


        mne_data.get_HRV(card_sig_list)
        mne_data.get_RSA(rsp_sig_list)
        mne_data.save_raws(raw_list) #save filtered eeg
        #clean variables
        del([mne_data,DF_class,annot,raw_list,rsp_sig_list,card_sig_list])
        #clean cache
        gc.collect()
        plt.close('all')
