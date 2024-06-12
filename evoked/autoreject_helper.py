# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 15:20:41 2021

@author: Engi
"""

from autoreject import AutoReject
import mne
import evoked.evoked_constants as ev_cs
import numpy as np
import base.files_in_out as in_out
import feather
class AutoHelp():
    def __init__(self,files):
        self.files=files
        self.label=['vep','hep','xns']
        
        #self.label=[['vep']]



    def get_epochs(self):
        
        self.epochs=mne.read_epochs(self.files)
        self.epochs_list=[]
        for lab in self.label:
          self.epochs_list.append(self.epochs[lab])


    def compute_autorej(self):
        self.clean_list=[]
        self.log_list=[]
        self.ar_list=[]
        for epoch in self.epochs_list:
       
            
            ar=AutoReject(random_state=42)
            ar.fit(epoch)
            epochs_clean=ar.transform(epoch)
            rej_log=ar.get_reject_log(epoch)
            

            self.clean_list.append(epochs_clean)
            self.log_list.append(rej_log)
            self.ar_list.append(ar)

    def save_output(self,files):

        for clean,log,ar,lab in zip(self.clean_list,self.log_list,self.ar_list,self.label):
            # type_sig_log='epochs/'+files.g_num+'_logs'
            type_sig_clean='epochs/'+files.g_num+'_final'
            file_end_epo=lab+'_clean_epo.fif'
            file_end_epo_eeglab=lab+'_clean_epo.set'
            # file_end_log=lab+'_log.png'
            # file_end_ar=lab+'_ar.h5'

            output_filename_epo=files.out_filename(type_sig=type_sig_clean,
                                                   file_end=file_end_epo)

            output_filename_epo_eeglab=files.out_filename(type_sig=type_sig_clean,
                                                    file_end=file_end_epo_eeglab)


            # output_filename_log=files.out_filename(type_sig=type_sig_log,
            #                                        file_end=file_end_log)
            #output_filename_ar=files.out_filename(type_sig=type_sig_log,
                                                   #file_end=file_end_ar)
            #save epochs
            clean.save(output_filename_epo,overwrite=True)
            clean.export(output_filename_epo_eeglab,overwrite=True)
            #save plot
            fig=log.plot()
            #fig.savefig(output_filename_log,dpi=1000)
            siz=np.where(log.bad_epochs==True)
            size_rej=np.size(siz)
            size_tot=np.size(log.bad_epochs)
            self.report.add_figure(fig, title=f'rej log for {lab}', caption=f'dropped {size_rej} out of {size_tot} epochs')
            #save AR
            #ar.save(output_filename_ar,overwrite=True)


    def get_lab(self,lab):
        if '/' in lab:
            end_ix=lab.split('/')
            suff=''.join(['_'+x for x in end_ix])
        else:
            suff='_'+lab

        return suff

    def check_type(self,epo_id):
        if 'vep' in epo_id[0]:
                list_id=ev_cs.id_vep
                cond_type='vep'
        if 'hep' in epo_id[0]:
                list_id=ev_cs.id_hep_fin
                cond_type='hep'
        if 'xns' in epo_id[0]:
                list_id=ev_cs.id_xns
                cond_type='xns'
        return list_id,cond_type

    def save_erps(self,out_filename,data):
        with open(out_filename,'w') as output:
            np.savetxt(output,np.column_stack(data),fmt='%1.10f')
        



                        

#convert to matlab format
class EpochConv():
    def __init__(self,filename,files): 
        self.files=files
        self.filename=filename

        self.epo=mne.read_epochs(self.filename)
    def save_eeglabformat(self):
        path=self.filename[:-3]
        file_name='set'
        out_filename=path+file_name
        self.epo.export(out_filename)
    
    def save_metadata(self):
        out_filename=self.files.out_filename(type_sig='mrk_DF',file_end='metadata_filt_rsp.feather')
        meta=self.epo.metadata
        feather.write_dataframe(meta,out_filename)
    