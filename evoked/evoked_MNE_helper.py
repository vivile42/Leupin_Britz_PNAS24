#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 13:37:15 2021

@author: leupinv
"""
import mne
import evoked.evoked_constants as ev_cs
import os
import numpy as np
import evoked.evoked_constants as cs

class EpochGroup():
    def __init__(self):
        self.vep_group = []
        self.hep_group = []
        self.xns_group = []


    def find_sys_med(self, epoch):
        epo = epoch['cardiac_phase=="sys"']
        meta = epo.metadata
        median = meta['R_stim_int'].median()
        print(median)
        return median

    def add_epoch(self, file, g_num, idx):
        epoch = mne.read_epochs(file)
        epoch.info['subject_info'] = dict(his_id=g_num)
        if '_n_' in file:
            self.tsk_cond='n'
        elif '_o_' in file:
            self.tsk_cond='o'

        else:
            raise ValueError('expected either o or n as condition')
        print(self.tsk_cond)

        print(epoch.info['subject_info'])
        #account for systolic mask


        self.sys_mask_all = ['sys_mask==1', 'noh']

        self.sys_mask = self.sys_mask_all[idx]

        sys_lab = ev_cs.sys_lab
        #Run with maskON maskOFF
        if self.sys_mask == 'noh':
           epo = epoch
        else:
            epo = epoch[self.sys_mask]
        self.sys_lab = sys_lab[idx]

        #add epoch, if looking at sys early vs late skip hep and xns

        if 'vep' in file:
            self.vep_group.append(epo)

        elif 'hep' in file:
            if any(x == self.sys_lab for x in ev_cs.only_vep):
                pass
            else:
                self.hep_group.append(epo)

        elif 'xns' in file:
            if any(x == self.sys_lab for x in ev_cs.only_vep):
                pass
            else:
                self.xns_group.append(epo)

        else:
            pass

    def check_type(self, epo_id):
        if 'vep' in epo_id[0]:
            list_id = ev_cs.id_vep
            cond_type = 'vep'
        if 'hep' in epo_id[0]:
            list_id = ev_cs.id_hep_fin
            cond_type = 'hep'
        if 'xns' in epo_id[0]:
            list_id = ev_cs.id_xns
            cond_type = 'xns'
        return list_id, cond_type

    def get_all_list(self):
        if any(x == self.sys_lab for x in ev_cs.only_vep):
            self.all_list = [self.vep_group]
        else:
            self.all_list = [self.vep_group, self.hep_group, self.xns_group]

    def get_averages(self, cfa, miss=False):
        #loop for each condition category

        diffi = ev_cs.diffi_list

        for epo_kind in self.all_list[:1]:

            accuracy = ev_cs.accuracy_cond
            for self.acc in accuracy:

                for dif in diffi:

                    self.grand_averages = []
                    epo_id = [x for x in epo_kind[0].event_id.keys()]
                    #epo_id=[x for x in epo_id if self.acc in x ]

                    list_id, self.cond_type = self.check_type(epo_id)
                    self.dif = dif

                    for lab in list_id:
                        self.lab = lab

            
                        self.group_evo_raw = []
                        if miss:
                            print('miss')
                            for epo in epo_kind:

                                epo = epo['awareness == "unaware" or accuracy == "correct"']
                                self.evo_raw = epo[dif+'/'+lab].average()
                                self.G_n = self.evo_raw.info['subject_info']['his_id']
                                self.evo_raw.comment = self.evo_raw.comment + \
                                    f"\n G_n= {self.G_n}"
                                self.save_single_erp(self.evo_raw, cfa,miss=True)
                                self.group_evo_raw.append(self.evo_raw)

                        else:
                            for epo in epo_kind:
                                self.evo_raw = epo[self.acc
                                                   + '/'+dif+'/'+lab].average()
                                self.G_n = self.evo_raw.info['subject_info']['his_id']
                                self.evo_raw.comment = self.evo_raw.comment + \
                                    f"\n G_n= {self.G_n}"
                                self.save_single_erp(self.evo_raw, cfa)
                                self.group_evo_raw.append(self.evo_raw)

                        self.grand_average = mne.grand_average(
                            self.group_evo_raw)
                        self.grand_average.comment = lab

                        self.grand_averages.append(self.grand_average)

                        # except:
                        #     print(self.sys_lab+' '+self.acc+' '+cfa+' '
                        #           + self.cond_type+' '+self.dif[:4]+' '+lab+' was skipped')
                        #     continue
                        if miss:
                            self.save_evoked(cfa,miss=True)
                            self.save_single_gavg(cfa,miss=True)
                            try:
                                self.save_gavg(cfa,miss=True)
                            except:
                                pass
                        else:
                            self.save_evoked(cfa)
                            self.save_single_gavg(cfa)
                            try:
                                self.save_gavg(cfa)
                            except:
                                pass


    def save_evoked(self, cfa,miss=False):
        if miss:
    
            filepath = 'ana/MNE/evo_list/'+self.sys_lab + \
                '/'+cfa+'/'+self.cond_type+'/'+self.dif[:4]+'/'
            filename = f'tsk_{self.tsk_cond}_'+self.sys_lab+'_'+cfa+'_'+self.cond_type + \
                '_'+self.dif[:4]+'_'+self.lab.replace('/', '_')+'_list-ave.fif'
        else:
            
            filepath = 'ana/MNE/evo_list/'+self.sys_lab+'/'+self.acc + \
                '/'+cfa+'/'+self.cond_type+'/'+self.dif[:4]+'/'
            filename = f'tsk_{self.tsk_cond}_'+self.sys_lab+'_'+self.acc+'_'+cfa+'_'+self.cond_type + \
                '_'+self.dif[:4]+'_'+self.lab.replace('/', '_')+'_list-ave.fif'
        print(filename)

        directory = filepath+filename

        if not os.path.exists(filepath):
            os.makedirs(filepath)

        mne.write_evokeds(directory, self.group_evo_raw)

    def save_gavg(self, cfa,miss=False):
        if miss:

            filepath = 'ana/MNE/gavg/'+self.sys_lab+'/'+self.acc+'/'+cfa+'/'
            filename = f'tsk_{self.tsk_cond}_'+self.sys_lab+'_'+self.acc+'_'+cfa+'_' + \
                self.cond_type+'_'+self.dif[:4]+'_'+'gavg-ave.fif'
        else:
            filepath = 'ana/MNE/gavg/'+self.sys_lab+'/'+self.acc+'/'+cfa+'/'
            filename = f'tsk_{self.tsk_cond}_'+self.sys_lab+'_'+cfa+'_' + \
                self.cond_type+'_'+self.dif[:4]+'_'+'gavg-ave.fif'
            

        directory = filepath+filename

        if not os.path.exists(filepath):
            os.makedirs(filepath)

        mne.write_evokeds(directory, self.grand_averages)

    def save_single_gavg(self, cfa,miss=False):
        if miss:

            filepath = 'ana/MNE/gavg/'+self.sys_lab+ \
                '/'+cfa+'/ep/'+self.cond_type+'/'+self.dif[:4]+'/'
            filename = f'tsk_{self.tsk_cond}_'+self.sys_lab+'_'+cfa+'_'+self.cond_type + \
                '_'+self.dif[:4]+'_'+self.lab.replace('/', '_')+'_gavg.ep'
            
        else:
            filepath = 'ana/MNE/gavg/'+self.sys_lab+'/'+self.acc + \
                '/'+cfa+'/ep/'+self.cond_type+'/'+self.dif[:4]+'/'
            filename = f'tsk_{self.tsk_cond}_'+self.sys_lab+'_'+self.acc+'_'+cfa+'_'+self.cond_type + \
                '_'+self.dif[:4]+'_'+self.lab.replace('/', '_')+'_gavg.ep'

        directory = filepath+filename

        if not os.path.exists(filepath):
            os.makedirs(filepath)

        data_ep = self.grand_average.data
        self.save_erps(directory, data_ep)

    def save_single_erp(self, evoked, cfa,miss=False):
        ep_data = evoked.data
        if miss:
            dir_erp = 'preproc/'+self.G_n+'/'+self.G_n+'_evoked'+'/'+self.G_n+'_'+self.sys_lab+'/' + self.G_n + \
                '_'+'_'+cfa+'/' + self.G_n + \
                '_'+self.cond_type+'/' + self.G_n+'_'+self.dif[:4]
    
            fileend = '/'+self.G_n+f'_{self.tsk_cond}_tsk_'+self.sys_lab+'_'+cfa+'_' + \
                self.cond_type+'_'+self.dif[:4]+'_' + \
                self.lab.replace('/', '_')+'.ep'
        else:
            dir_erp = 'preproc/'+self.G_n+'/'+self.G_n+'_evoked'+'/'+self.G_n+'_'+self.sys_lab+'/' + self.G_n + \
                '_'+self.acc+'/'+self.G_n+'_'+cfa+'/' + self.G_n + \
                '_'+self.cond_type+'/' + self.G_n+'_'+self.dif[:4]
    
            fileend = '/'+self.G_n+f'_{self.tsk_cond}_tsk_'+self.sys_lab+'_'+self.acc+'_'+cfa+'_' + \
                self.cond_type+'_'+self.dif[:4]+'_' + \
                self.lab.replace('/', '_')+'.ep'
        filename = dir_erp+fileend
        if not os.path.exists(dir_erp):
            os.makedirs(dir_erp)
        self.save_erps(filename, ep_data)
        print(f'writing single erp: {fileend} ')

    def save_erps(self, out_filename, data):
        with open(out_filename, 'w') as output:
            np.savetxt(output, np.column_stack(data), fmt='%1.10f')
