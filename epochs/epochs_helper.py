#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 10:38:57 2021

Epochs helping function
@author: leupinv
"""

import pandas as pd
import numpy as np
import mne
from mne.preprocessing import ICA
import base.files_in_out as files_in_out
import feather
import epochs.epochs_constants as cs



def find_nearest(a, a0):
    idx = np.abs(a - a0).argmin()
    return a.flat[idx]


class Epoch_HP():
    def __init__(self,files):
        self.files=files
        self.raw=mne.io.read_raw_fif(self.files.current_file_dir,preload=True)
        #self.get_bad_interval()
        self.events_from_annot, self.event_dict = mne.events_from_annotations(self.raw)
        self.init_report()
        #self.run_ICA()

    def init_report(self):

        #self.report.parse_folder('raw', pattern=f'{self.files.g_num}*{self.files.condition}*.fif',
                                 #render_bem=False,verbose=True)
        type_sig='report'
        file_end='report.h5'

        filename=self.files.out_filename(type_sig=type_sig,file_end=file_end)
        print(filename)
        self.report = mne.open_report(filename)
    def save_report(self):
        type_sig='report'
        file_end='report.html'

        filename=self.files.out_filename(type_sig=type_sig,file_end=file_end)
        self.report.save(filename, overwrite=True)

    def get_metadata(self):
        files_meta = files_in_out.GetFiles(filepath='preproc',eeg_format='off',
                                           condition=self.files.condition,
                                           g_num=self.files.g_num)
        files_meta.filter_file(filters='metadata.feather')
        metadata=pd.read_feather(files_meta.filt[0])
        return metadata






    def run_infoICA(self):
        random_state = 42
        method='infomax'
        fit_params=dict(extended=True)
        self.ica = ICA(random_state=random_state,method=method,fit_params=fit_params,max_iter='auto')
        self.ica.fit(self.raw)

        #save ICA
        type_sig='ICA'

        file_end='info-ica.fif'
        output_filename=self.files.out_filename(type_sig=type_sig,file_end=file_end)

        self.ica.save(output_filename)

    def run_fastICA(self):
        random_state = 42
        self.runica = ICA(random_state=random_state,max_iter='auto')
        self.runica.fit(self.raw,decim=2)

        #save ICA
        type_sig='ICA'

        file_end='fast-ica.fif'
        output_filename=self.files.out_filename(type_sig=type_sig,file_end=file_end)
        self.runica.save(output_filename)
    def isica(self):
        try:
            self.read_ICA()
            self.read_ICA_log(write=True)

            return True
        except:

            return False



    def select_ICA_components(self):
        self.read_ICA()
        self.read_ICA_log()

        #plot components to exclude
        arg=dict(vmax=2,vmin=-2)
        figs_ecg=self.ica.plot_properties(inst=self.ecg_epochs,picks=self.dict_el['ecg_index'],image_args=arg)
        figs_eog=self.ica.plot_properties(inst=self.eog_epochs,picks=self.dict_el['eog_index'],image_args=arg)

        self.report.add_figs_to_section(figs_ecg,
                           captions=[ f'ECG component -{x+1}' for x,_ in enumerate(figs_ecg)],section='ICA')
        self.report.add_figs_to_section(figs_eog,
                           captions=[ f'EOG component -{x+1}' for x,_ in enumerate(figs_eog)],section='ICA')
        #  get epochs with CFA (eclude EOG + artefacts)

        self.ica.exclude.extend(self.dict_el['eog_index']+self.dict_el['artefact_index'])

        self.epo_cfa=self.epochs_exp.copy()



        self.ica.apply(self.epo_cfa)
        #Exclude ECG --> NO CFA epochs

        self.ica.exclude.extend(self.dict_el['ecg_index'])



        self.epo_nc=self.epochs_exp.copy()

        self.ica.apply(self.epo_nc)
        # save epochs

        self.save_epoch_ICA()


    def save_epoch_ICA(self):
        type_sig='epochs'
        file_end='nc_rec_epo.fif'

        output_filename=self.files.out_filename(type_sig=type_sig,file_end=file_end)

        self.epo_nc.save(output_filename,overwrite=True)


        file_end='cfa_rec_epo.fif'

        output_filename=self.files.out_filename(type_sig=type_sig,file_end=file_end)

        self.epo_cfa.savemma(output_filename,overwrite=True)

    def read_ICA(self):
        type_sig='ICA'
        file_end='fast-ica.fif'

        filename=self.files.out_filename(type_sig=type_sig,file_end=file_end)
        self.ica=mne.preprocessing.read_ica(filename)





    def read_ICA_log(self,write=False):
        import ast
        type_sig='ICA'


        if write==False:

            file_end='ICA_log.txt'
            filename=self.files.out_filename(type_sig=type_sig,file_end=file_end)
            print(filename)

            with open(filename) as log:
                self.dict_el={}
                for line in log:
                    key,value=line.split('=')
                    key=key.replace(' ','_')
                    value=ast.literal_eval(value)
                    self.dict_el[key]=value
        else:
            file_end='ICA_log_missing.txt'
            filename=self.files.out_filename(type_sig=type_sig,file_end=file_end)
            with open(filename,'w') as log:
                message=f"{self.files.g_num} doesn't have yet log"
                log.write(message)
                print(message)









    def get_eog_epochs(self,ch_name=['C16'],thresh=None):
         #self.raw.set_annotations(self.annotations_bad)

         self.eog_epochs=mne.preprocessing.create_eog_epochs(self.raw,ch_name=ch_name,
                                                        baseline=(-0.5,-0.2),thresh=thresh)

         visual_range={ k:v for k,v in self.event_dict.items() if 'vep' in k} #dict index for the VEPs events

         visual_events=[ev[0] for ev in self.events_from_annot if ev[2] in list(visual_range.values())]
         visual_events=np.array(visual_events)
         eog_events=self.eog_epochs.events


         onset = [blink[0]/self.raw.info['sfreq']-0.5 for blink in eog_events if (blink[0] - find_nearest( visual_events, blink[0] )<75) and (blink[0] - find_nearest( visual_events, blink[0] )>-75)]

         n_blink=len(eog_events)

         # if len(onset) == 0:
         #     onset=[1,2,3,4]


         n_blinks = len(onset)
         print(f'number of bad blinks : {n_blinks}')


         duration = np.repeat(0.5, n_blinks)
         description = ['bad blink'] * n_blinks



        #save eog log
         type_sig='epochs'

         file_end='eog_log.txt'
         output_filename=self.files.out_filename(type_sig=type_sig,file_end=file_end)


         with open(output_filename,"w") as file:
            file.write(f'number of blinks:{n_blink} and of bad blinks:{n_blinks}\n')
            for line in zip(onset,description):
                file.write(f'{line}\n')



         self.raw.annotations.append(onset, duration, description)

         eog_fig=self.eog_epochs.average().plot_joint()
         self.report.add_figure(eog_fig,title='blinks',caption=f'number of blinks:{n_blink} and of bad blinks:{n_blinks}\n')


    def get_exp_epochs(self):
         TF=[tf[0] for  tf in self.events_from_annot]
         tf=pd.Series(TF)
         print(np.where(tf.duplicated()==True))
         while not tf[tf.duplicated()].empty:
             dup=tf[tf.duplicated()]
             dup=dup.index
             for du in dup:

                 self.events_from_annot[du][0]-=1

             TF=[tf[0] for  tf in self.events_from_annot]
             tf=pd.Series(TF)


         ## define epochs



         picks = mne.pick_types(self.raw.info, meg=False, eeg=True)

         # Add metadata
         metadata=self.get_metadata()
         self.epochs_exp = mne.Epochs(self.raw, self.events_from_annot, self.event_dict, cs.tmin, cs.tmax, proj=True,
                            picks=picks, baseline=cs.baseline, reject=None, metadata=metadata,
                            preload=True,verbose=True)
         fig_drop_log=self.epochs_exp.plot_drop_log()
         psd_fig=self.epochs_exp.plot_psd()

         self.report.add_figure([fig_drop_log,psd_fig],
                           title=['Dropped Epochs','PSD Epoched'])

    def get_ecg_epochs(self):
        self.ecg_epochs=mne.preprocessing.create_ecg_epochs(self.raw)
        ecg_fig=self.ecg_epochs.average().plot_joint()
        self.report.add_figure(ecg_fig,title='heartbeats')

    def save_epochs(self):
        type_sig='epochs'

        file_end='eog_epo.fif'
        output_filename=self.files.out_filename(type_sig=type_sig,file_end=file_end)

        self.eog_epochs.save(output_filename,overwrite=True)

        file_end='exp_epo.fif'
        output_filename=self.files.out_filename(type_sig=type_sig,file_end=file_end)

        self.epochs_exp.save(output_filename,overwrite=True)



        file_end='ecg_epo.fif'
        output_filename=self.files.out_filename(type_sig=type_sig,file_end=file_end)

        self.ecg_epochs.save(output_filename,overwrite=True)
