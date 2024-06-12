#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 09:38:22 2021

@author: leupinv
"""
import mne
import source.source_constants as cs
import os

def check_dir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)

def check_type(check_obj,list_type):
    for check in list_type:
        if check in check_obj:
            return check
    return None





class GetCovariance():
    def __init__(self,file,g_num,info_raw,loose=True,cov_end=True):
        self.g_num=g_num
        self.epoch=mne.read_epochs(file)
        self.info_raw=info_raw
        self.source_ori=cs.source_ori
        self.cov_end_cond=cs.cov_end_cond
        self.filepath=cs.dir_fold+f'/ana/MNE/source/{cs.n_sources}_source/{self.source_ori}/{self.cov_end_cond}/depth_{cs.inv_depth}/'
        self.get_cond()
        self.get_cfa(file)


    def get_cond(self):
        list_type=['vep']
        epo_id=[x for x in self.epoch.event_id.keys()]
        self.cond_type=check_type(epo_id[0],list_type)
    def get_cfa(self,file):
        list_type=['cfa','nc']
        self.cfa_type=check_type(file,list_type)



    def get_cov(self):

        self.cov=mne.compute_covariance(self.epoch, method=cs.cov_method,
                                              tmin=cs.cov_tmin,rank=None)

        ## save covariance
        cov_path=self.filepath+f'cov/{self.cfa_type}/{self.cond_type}/'
        cov_name=f'{self.g_num}_{self.cfa_type}_{self.cond_type}-cov.fif'


        check_dir(cov_path)
        filename=cov_path+cov_name


        self.cov.save(filename)


    def get_inverse(self):
        fwd=mne.read_forward_solution(cs.fwd_dir)
        if self.source_ori == 'loose':
            inv_loose= 1.
        elif self.source_ori =='fixed':
            inv_loose=0.
        inv = mne.minimum_norm.make_inverse_operator(self.info_raw, fwd, self.cov, loose=inv_loose, depth=cs.inv_depth,
                            verbose=True)
        #save inverse
        inv_path=self.filepath+f'inv/{self.cfa_type}/{self.cond_type}/'
        inv_name=f'{self.g_num}_{self.cfa_type}_{self.cond_type}-inv.fif'



        check_dir(inv_path)
        filename=inv_path+inv_name


        mne.minimum_norm.write_inverse_operator(filename,inv)






class GetSTC():
    # source time course, apply inverse operator to erp to get the source
    def __init__(self,file_inv,inv_obj,file_evo,g_num,loose=True,cov_end=True,method='eLORETA'):
        self.g_num=g_num
        if loose:
            self.source_ori='loose'
        else:
            self.source_ori='fixed'
        if cov_end:
            self.cov_end_cond='cov_end'
        else:
            self.cov_end_cond='cov_start'
        self.inv_method=method

        self.filepath=cs.dir_fold+f'/ana/MNE/source/{cs.n_sources}_source/{self.source_ori}/{self.cov_end_cond}/depth_{cs.inv_depth}/stc/{self.inv_method}/'
        self.cfa_inv=self.get_cfa(file_inv)
        self.cfa_evo=self.get_cfa(file_evo)
        self.cond_inv=self.get_cond(file_inv)
        self.cond_evo=self.get_cond(file_evo)

        if self.check_continue(file_evo):
            print(file_inv)
            print(file_evo)
            try:
                self.evo=mne.read_evokeds(file_evo,verbose=False)
                self.evo=[x for x in self.evo if g_num in x.comment][0]
                self.inv=inv_obj
                self.get_out_filename(file_evo)
                self.apply_inverse_sol()
                self.save_stc()
            except:
                pass






    def get_cond(self,file):
        list_type=['vep','hep','xns']
        cond_type=check_type(file,list_type)
        return cond_type

    def get_cfa(self,file):
        list_type=['cfa','nc']
        cfa_type=check_type(file,list_type)
        return cfa_type


    def check_continue(self,file_evo):

        if check_type(file_evo,cs.sys_lab) == None:
            return False
        elif check_type(file_evo,["mistake","hep","maskON","maskOFF",'xns']) != None:
            return False

        elif self.cfa_inv==self.cfa_evo and self.cond_evo== self.cond_inv:
            return True
        else:
            return False

    def apply_inverse_sol(self):
        self.stc = mne.minimum_norm.apply_inverse(self.evo, self.inv, cs.lambda2,
                              method=self.inv_method, pick_ori=None, verbose=True)

    def get_out_filename(self,file_evo):
        list_dir=file_evo.replace("\\","/").split('/')
        dir_out=list_dir[3:-1]
        filename=list_dir[-1]
        dir_sup=filename.split('_')
        dir_sup=dir_sup[5:-1]
        dir_sup=os.path.join(*dir_sup)
        dir_out=os.path.join(*dir_out)
        dir_out=os.path.join(self.filepath,dir_out,dir_sup)
        filename=f'{self.g_num}_n_tsk_'+filename[:-13]
        out_filename=dir_out+'/'+filename
        check_dir(dir_out)
        self.out_filename=out_filename
    def save_stc(self):

        self.stc.save(self.out_filename)
































