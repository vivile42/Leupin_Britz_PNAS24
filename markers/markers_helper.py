#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 13:08:00 2021
marker files helping functios
@author: leupinv
"""
import markers.markers_constants as cs
from itertools import product

import pandas as pd
import feather
import mne
import numpy as np


def get_DF_centerd(df,cent_column,srate,method='backfill',RT=True):
    """fill in df so that only TFs relative to the referenced column are kept,
    method= backfill should be used for stimulus events
    method= pad should be used for responses'
    
    """
    len_columns=len(df.columns)
    mrk_columns=df.columns[1:len_columns]
    n=0

    df.sort_values('TF',inplace=True)
    
    if isinstance(method,list):
        for col in mrk_columns:
            if col!=cent_column:
                df[col]=df[col].fillna(method=method[n])
                n+=1
        
    else:
        for col in mrk_columns:
            if col!=cent_column:
                df[col]=df[col].fillna(method=method)
        
    if RT:
        # compute RT if argument true 
        df['RT']=df.TF.diff()
        df['RT']=df['RT'].shift(-1)
        df['RT']=(df['RT']/srate*1000)-8
    
    df=df[df[cent_column].notnull()]

    
    return df
        

def get_help_DF(df,columns_existing,new_column=None,new_column_name=None):
    """Given the name of the columns containing the variable to be combined returns either
    -combinatory table to help determine condition (new column and column name remains to none)
    -pandas dataframe to use has helper if list of new conditions is give"""
    
    
    list_value=[]
    for col in columns_existing:
        uniquedf=df[col].unique().tolist()
        uniquedf.sort()
        list_value.append(uniquedf)
        
    iteration=list(product(*list_value))    
    list_dict=[]
    for el in iteration:
        n_col=len(columns_existing)
        list_dict.append({(columns_existing[i]):(el[i]) for i in range(n_col)})
    list_df=pd.DataFrame(list_dict)
    if new_column_name!= None:     
        list_df[new_column_name]=new_column

    
    return list_df

def compute_mrk(df,cent_column,srate,method):
    mrk_df=get_DF_centerd(df.copy(),cent_column=cent_column,srate=srate,method=method)
    mrk_df=mrk_df.dropna(how='any',axis=0)
    mrk_df['difficulty']=['easy' if x==20 or x==10 else 'normal'for x in mrk_df['trigger_stim']]
    mrk_df['accuracy']=['correct' if x==3 else 'mistake' for x in mrk_df['trigger_corr']] 
    mrk_df['awareness']=['aware' if x==5 else 'unaware' for x in  mrk_df['trigger_aware']] 
    
    return mrk_df


def fix_merged(df):
    df.sort_values(by=['TF'],inplace=True)
    #df.set_index('TF',inplace=True)
    df['trigger_stim']=df['trigger_stim'].fillna(0)
    df['trigger_corr']=df['trigger_corr'].fillna(0)
    df['trigger_aware']=df['trigger_aware'].fillna(0)
    df['RT']=df['RT'].fillna(method='backfill')
    df['difficulty']=df['difficulty'].fillna(method='backfill')
    df['accuracy']=df['accuracy'].fillna(method='backfill')
    df['awareness']=df['awareness'].fillna(method='backfill')
   
    return df

def fix_merged_heart(df):
    df=fix_merged(df)
    df['cardiac_peak']=df['cardiac_phase'].fillna('no_peak')
    df['cardiac_phase']=df['cardiac_phase'].fillna(method='pad')
    df['cardiac_phase']=df['cardiac_phase'].map({'R':'sys','T':'dia'})
    df['cardiac_phase'].fillna(value='no_peak',inplace=True)
    

    columns=['RRI','HeartRate','HeartRate_precedent','HeartRate_post','HeartRate_rolling_5_before',
                  'HeartRate_rolling_5_after','HeartRate_rolling_5_centered','HeartRateVar_rolling_10_centered',
                  'HeartRateVar_rolling_100_centered']
    df[columns]=df[columns].fillna(method='pad')
    df[columns]=df[columns].fillna(method='backfill')
    
    return df

def fix_merged_rsp(df,rsp_df_mrk,stim=False):
    df=fix_merged(df)
    df['cardiac_peak']=df['cardiac_peak'].fillna('no_peak')
    df.sort_values(by=['TF'],inplace=True)
    
    df['rsp_phase']=df['rsp_phase'].fillna(method='pad')
    if rsp_df_mrk['rsp_phase'][0]=='inh':
        df['rsp_phase'].fillna(value='exh',inplace=True)
    else:
        df['rsp_phase'].fillna(value='inh',inplace=True)
    
    if stim:
        columns_0=['ECG_R_Peaks','ECG_T_Offsets'] 
        df['sys_mask']=df['sys_mask'].fillna(0)
        df['R_stim_int']=[x if s !=0 else 0 for x,s in zip(df['R_stim_int'],df['trigger_stim'])]
        df_filt=get_inh_stim_int(df)
        df=df.merge(df_filt,on='TF',how='outer')
        df['inh_stim_int'].fillna(0,inplace=True)
    else:
        columns_0=['ECG_R_Peaks','ECG_T_Peaks'] 
    df.drop(columns_0,axis=1,inplace=True)
    
    
    columns=['RRI','cardiac_phase','HeartRate','HeartRate_precedent','HeartRate_post','HeartRate_rolling_5_before',
                  'HeartRate_rolling_5_centered','HeartRate_rolling_5_after','HeartRateVar_rolling_10_centered',
                  'HeartRateVar_rolling_100_centered','rsp_int','RSP_Rate','RSP_Rate_post','RSP_Rate_precedent','RSP_Rate_rolling_5_before',
                  'RSP_Rate_rolling_5_centered','RSP_Rate_rolling_5_after','RspRateVar_rolling_10_centered',
                  'RspRateVar_rolling_100_centered']
    df[columns]=df[columns].fillna(method='pad')
    df[columns]=df[columns].fillna(method='backfill')
    
    return df
    

    

def get_duration_phase(df):
    df['phase_duration']=df.TF.diff().shift(-1)

    return df
    
def get_inh_stim_int(df):
    df_filt=df[['TF','rsp_phase']].loc[(df['cardiac_peak']=='no_peak')&(df['trigger_stim']==0)&(df['rsp_phase']=='inh')]
    df_stim=df[['TF','trigger_stim']].loc[df['trigger_stim']!=0]
    df_filt['TF_rsp']=df_filt['TF']
    df_stim['TF_stim']=df_stim['TF']

    filt=df_stim.merge(df_filt,how='outer',on='TF')
    filt.sort_values(by=['TF'],inplace=True)
    filt['TF_rsp'].fillna(method='pad',inplace=True)
    filt['inh_stim_int']=filt['TF_stim']-filt['TF_rsp']
    
    filt=filt[['TF','inh_stim_int']][filt['rsp_phase']!='inh']
    
    return filt
    

class DF_Markers():
    def __init__(self,MNE_mrk):
        self.MNE_mrk=MNE_mrk
        self.mrk_df=MNE_mrk.mrk
        self.srate=MNE_mrk.srate
        self.rsp_DF=MNE_mrk.rsp_df_mrk
        self.ecg_DF=MNE_mrk.cardiac_mrk_stim
        self.hep_DF=MNE_mrk.cardiac_mrk_Tpk
        self.duration_rsp_ecg()
        self.files=MNE_mrk.files
        self.compute_stim_mrk()
        self.compute_ans_mrk()
        self.merge_mrk_DF()
        
        self.format_merged()
        self.compute_cartool_markers()
        self.compute_heps_mrk()
        self.save_DF()
        ##self.filter_conditions()
        self.save_cartool_markers()
       

        
    
         
    def compute_stim_mrk(self):
        self.stim_DF=compute_mrk(self.mrk_df,'trigger_stim',self.srate,method='backfill')
        
    def compute_ans_mrk(self):
        self.ans_DF=compute_mrk(self.mrk_df,'trigger_corr',self.srate,method=cs.method_ans)
    def duration_rsp_ecg(self):
        get_duration_phase(self.rsp_DF)
        get_duration_phase(self.ecg_DF)
        get_duration_phase(self.hep_DF)
    def check_dup(self,df):
        dup=df['TF'][df.TF.duplicated()]
        
        print(dup)
        
        if not dup.empty:
            dup=dup.index
            df.loc[dup,'TF']-=1
        return df
            
        
    def merge_mrk_DF(self):
        #stim
        self.stim_df=pd.concat([self.stim_DF,self.ecg_DF])
        self.stim_df=self.check_dup(self.stim_df)
        self.stim_df=self.compute_sys_interval(self.stim_df)
        self.stim_df=pd.concat([self.stim_df,self.rsp_DF])
        self.stim_df=self.check_dup(self.stim_df)
        #heps
        self.heps_mrk=pd.concat([self.stim_DF,self.hep_DF])
        self.heps_mrk=self.check_dup(self.heps_mrk)
        self.heps_mrk=fix_merged_heart(self.heps_mrk)
        self.heps_mrk=pd.concat([self.heps_mrk,self.rsp_DF])
        self.heps_mrk=self.check_dup(self.heps_mrk)
        


        
        
        #ans marker centered on the answer 
        self.ans_mrk_ans=pd.concat([self.ans_DF,self.ecg_DF])
        #
        self.ans_mrk_ans=self.compute_sys_interval(self.ans_mrk_ans)
        self.ans_mrk_ans=pd.concat([self.ans_mrk_ans,self.rsp_DF])
    
    def compute_sys_interval(self,df):
        df=fix_merged_heart(df)
        self.df_filt=self.compute_R_stim_int(df)
        df['TF_shift']=df['TF'].shift(-1)
        df['phase_duration']=df.apply(
            lambda row : row['TF_shift']-row['TF'] if np.isnan(row['phase_duration']) else row['phase_duration'],axis=1)
     
        df['sys_mask']=[1 if x=='no_peak' and y <= z else 0 for x,y,z in zip(df['cardiac_peak'],df['phase_duration'],df['phase_duration'].shift(2)) ]
        df['sys_mask']=[1 if x=='sys' and z!=0  else y for x,y,z in zip (df['cardiac_phase'],df['sys_mask'],df['trigger_stim'])]
               
        df=df.merge(self.df_filt,on='TF',how='left')
        
        df.drop('TF_shift',axis=1,inplace=True)
        return df
    def compute_R_stim_int(self,df):
        df_filt=df.loc[df['cardiac_peak']!='T']
        df_filt['R_stim_int']=(df_filt['TF'].diff())/self.srate
        df_filt=df_filt[['TF','R_stim_int']]
        return df_filt
    
    def format_merged(self):
        #stim
        self.stim_df=fix_merged_rsp(self.stim_df,self.rsp_DF,stim=True)

    # def place_holder(self):
        
        #save complete DF for stim
        outputfilename=self.files.out_filename(type_sig=cs.type_sig_mrk_DF,file_end='stim_complete_df'+cs.file_end_feather,short=True)
        feather.write_dataframe(self.stim_df,outputfilename) 
        self.stim_mrk=self.stim_df.dropna()

        
        
        self.stim_mrk=self.stim_mrk[self.stim_mrk['trigger_corr']!=0]
        self.stim_mrk.drop(['trigger_stim', 'trigger_corr', 'trigger_aware'],axis=1,inplace=True)
        
        #heps                
        self.heps_mrk=fix_merged_rsp(self.heps_mrk,self.rsp_DF)
        #hep
        self.hep_mrk=self.heps_mrk.loc[(self.heps_mrk['cardiac_peak']!='no_peak')|(self.heps_mrk['trigger_corr']!=0)]
        self.hep_mrk=self.hep_mrk[(self.hep_mrk['trigger_corr']!=0).shift(-1).fillna(False)]
        self.hep_mrk.drop(['trigger_stim', 'trigger_corr', 'trigger_aware'],axis=1,inplace=True)
        #self.hep_mrk.drop('index',axis=1,inplace=True)
        
        #hep2
        self.hep2_mrk=self.heps_mrk.loc[(self.heps_mrk['cardiac_peak']!='no_peak')|(self.heps_mrk['trigger_corr']!=0)]
        self.hep2_mrk=self.hep2_mrk[(self.hep2_mrk['trigger_corr']!=0).shift(-2).fillna(False)]
        #self.hep2_mrk.reset_index(inplace=True)
        self.hep2_mrk.drop(['trigger_stim', 'trigger_corr', 'trigger_aware'],axis=1,inplace=True)
        self.hep2_mrk['cardiac_peak']=self.hep2_mrk['cardiac_peak'].map({"R":"R2","T":"T2"})
        

        
        #RR
        self.RR_mrk=self.stim_df.loc[(self.stim_df['cardiac_peak']=='R')|(self.stim_df['trigger_corr']!=0)]
        self.RR_mrk= self.RR_mrk[( self.RR_mrk['trigger_corr']!=0).shift(-1).fillna(False)]
        self.RR_mrk['cardiac_peak']='RR'
        self.RR_mrk.drop(['trigger_stim', 'trigger_corr', 'trigger_aware'],axis=1,inplace=True)
        #self.RR_mrk.reset_index(inplace=True)

        #ans
        #ans
        self.ans_DF['ans']=1
        self.ans_mrk=pd.concat([self.stim_mrk,self.ans_DF])
        self.ans_mrk.sort_values(by=['TF'],inplace=True)
        self.ans_mrk['ans'].fillna(0,inplace=True)
        self.ans_mrk.fillna(method='pad',inplace=True)
        self.ans_mrk=self.ans_mrk[self.ans_mrk['ans']==1]
        self.ans_mrk.drop('ans',axis=1,inplace=True)
        self.ans_mrk.drop(['trigger_stim', 'trigger_corr', 'trigger_aware'],axis=1,inplace=True)
        self.ans_mrk.reset_index(inplace=True)
        self.ans_mrk.drop('index',axis=1,inplace=True)
        self.ans_mrk.reset_index(inplace=True)       
        self.ans_mrk.rename(dict(index='stim_idx'),axis=1,inplace=True)
        
        #ans marker centered on the answer 

        self.ans_mrk_ans=fix_merged_rsp(self.ans_mrk_ans,self.rsp_DF,stim=True)
        self.ans_mrk_ans.dropna(inplace=True)
  
        self.ans_mrk_ans=self.ans_mrk_ans[self.ans_mrk_ans['trigger_corr']!=0]
        self.ans_mrk_ans.drop(['trigger_stim', 'trigger_corr', 'trigger_aware'],axis=1,inplace=True)
        
        outputfilename=self.files.out_filename(type_sig=cs.type_sig_mrk_DF,file_end='ans_phase_on_ans.df'+cs.file_end_feather,short=True)
        feather.write_dataframe(self.ans_mrk_ans,outputfilename) 
    
    def compute_cartool_markers(self):
        mrk_DF_list=[self.stim_mrk,self.ans_mrk]
        hep_DF_list=[self.hep_mrk,self.hep2_mrk,self.RR_mrk]
        for DF in mrk_DF_list:            
            DF['mrk_awa']=[f'{x[0].upper()}{y[0].upper()}' for x,y in zip(DF['accuracy'],DF['awareness'])]
            DF['mrk_card_awa']=[f'{x}{y[0].upper()}' for x,y in zip(DF['mrk_awa'],DF['cardiac_phase'])]
            DF['mrk_rsp_awa']=[f'{x}{y[0].upper()}' for x,y in zip(DF['mrk_awa'],DF['rsp_phase'])] 
        for DF in hep_DF_list:
            DF['mrk_awa']=[f'{x[0].upper()}{y[0].upper()}' for x,y in zip(DF['accuracy'],DF['awareness'])]
            DF['mrk_card_awa']=[f'{x}{y}' for x,y in zip(DF['cardiac_peak'],DF['mrk_awa'])]
            DF['mrk_rsp_awa']=[f'{x}{y[0].upper()}{z}' for x,y,z in zip(DF['cardiac_peak'],DF['rsp_phase'],DF['mrk_awa'])]
    
    def compute_heps_mrk(self):
        heps_mrk=pd.concat([self.hep_mrk,self.hep2_mrk])
        heps_mrk.sort_values(by='TF',inplace=True)
        heps_mrk.drop('phase_duration',axis=1,inplace=True)
        self.RR_mrk.drop('phase_duration',axis=1,inplace=True)
        
        self.heps_df_mrk=heps_mrk.merge(self.RR_mrk,on=['TF','RT','difficulty','accuracy','awareness','mrk_awa','rsp_phase','cardiac_phase','RRI',
                                                        'HeartRate','HeartRate_precedent','HeartRate_post','HeartRate_rolling_5_before',
                  'HeartRate_rolling_5_centered','HeartRate_rolling_5_after','HeartRateVar_rolling_10_centered',
                  'HeartRateVar_rolling_100_centered','rsp_int','RSP_Rate','RSP_Rate_post','RSP_Rate_precedent','RSP_Rate_rolling_5_before',
                  'RSP_Rate_rolling_5_centered','RSP_Rate_rolling_5_after','RspRateVar_rolling_10_centered',
                  'RspRateVar_rolling_100_centered'],how='left',suffixes=["_hep","_RR"])
        
        self.heps_df_mrk.sort_values(by=['TF'],inplace=True)                                                                                
        self.heps_df_mrk.cardiac_peak_hep.fillna('RR',inplace=True)
        col_fill=['mrk_card_awa_hep', 'mrk_rsp_awa_hep', 'cardiac_peak_RR','mrk_card_awa_RR', 'mrk_rsp_awa_RR']
        for col in col_fill:
            self.heps_df_mrk[col]=self.heps_df_mrk[col].fillna('no_val')
        
        self.heps_df_mrk.drop(['cardiac_peak_RR'],axis=1,inplace=True)
    
    def merge_combined_df(self):
        heps=self.heps_df_mrk.copy()
        heps.drop('sys_mask',axis=1,inplace=True)
        vep=self.stim_mrk.copy()
        xns=self.ans_mrk.copy()
        vep.drop('phase_duration',axis=1,inplace=True)
        xns.drop('phase_duration',axis=1,inplace=True)
        # prepare heps to merge
        heps=heps.rename(columns=dict(mrk_card_awa_hep='mrk_card_awa',
                                 mrk_rsp_awa_hep='mrk_rsp_awa',
                                 cardiac_peak_hep='cardiac_peak'))
        heps['signal_type']='hep'
        # prepare vep + xns to merge
        vep['signal_type']='vep'
        xns['signal_type']='xns'
        
        merged=pd.concat([heps,vep,xns])
        merged.sort_values('TF', inplace=True)
        merged.sys_mask.fillna(method='backfill',inplace=True)
        merged.stim_idx.fillna(method='backfill',inplace=True)
        merged.R_stim_int.fillna(0,inplace=True)
        merged['inh_stim_int'].fillna(0,inplace=True)
        merged.fillna('no_val',inplace=True)
        self.merged=merged.copy()
        return merged
        
    def get_metadata(self):         
        # get cardiac df and merge it with stim to insert phase info
        card=self.MNE_mrk.ecg_signals.copy()
        card.reset_index(inplace=True)
        card.rename(columns=dict(index='TF'),inplace=True)
        card['ECG_Phase_Completion_Ventricular']=[(1-val)*-1 if phase ==1 else val 
                                                  for val,phase in zip(card['ECG_Phase_Completion_Ventricular'],
                                                                   card['ECG_Phase_Ventricular']) ]
        card['ECG_Phase_Completion_Atrial']=[(1-val)*-1 if phase ==1 else val 
                                                  for val,phase in zip(card['ECG_Phase_Completion_Atrial'],        
                                                                       card['ECG_Phase_Atrial']) ]
        
        card=card[['TF', 'ECG_Phase_Completion_Ventricular','ECG_Phase_Completion_Atrial']]
        
        self.merge_combined_df()
        merge_card=self.merged.merge(card,on='TF')
        
        # Do the same with resp
        resp=self.MNE_mrk.rsp_signals.copy()
        resp.reset_index(inplace=True)
        resp.rename(columns=dict(index='TF'),inplace=True)
        resp['RSP_Phase_Completion']=[(1-val)*-1 if phase ==1 else val 
                                                  for val,phase in zip(resp['RSP_Phase_Completion'],
                                                                  resp['RSP_Phase']) ]
        resp=resp[['TF','RSP_Phase_Completion','RSP_Amplitude']]
        self.fin_merge=merge_card.merge(resp,on='TF')
        self.fin_merge['inh_stim_int']=(self.fin_merge['inh_stim_int'])/self.srate
        outputfilename=self.files.out_filename(type_sig=cs.type_sig_mrk_DF,file_end='metadata'+cs.file_end_feather,short=True)
        feather.write_dataframe(self.fin_merge,outputfilename)  
        
        
    def save_DF(self):
        #Stim
        outputfilename=self.files.out_filename(type_sig=cs.type_sig_mrk_DF,file_end='stim_mrk'+cs.file_end_feather,short=True)
        feather.write_dataframe(self.stim_mrk,outputfilename)        
        #Heps       
        outputfilename=self.files.out_filename(type_sig=cs.type_sig_mrk_DF,file_end='heps_mrk'+cs.file_end_feather,short=True)
        feather.write_dataframe(self.heps_df_mrk,outputfilename) 
        
        #ans
        outputfilename=self.files.out_filename(type_sig=cs.type_sig_mrk_DF,file_end='ans_mrk'+cs.file_end_feather,short=True)
        feather.write_dataframe(self.ans_mrk,outputfilename)  
    def filter_conditions(self):
         #self.stim_mrk=self.stim_mrk[self.stim_mrk['difficulty']=='normal']
         self.stim_mrk=self.stim_mrk[self.stim_mrk['accuracy']=='correct']
         
         #self.heps_df_mrk=self.heps_df_mrk[self.heps_df_mrk['difficulty']=='normal']
         self.heps_df_mrk=self.heps_df_mrk[self.heps_df_mrk['accuracy']=='correct']
         
         #self.ans_mrk=self.ans_mrk[self.ans_mrk['difficulty']=='normal']
        
    def save_cartool_markers(self):
        #save text cartool markers in case we need to perform analyses using this software
        type_sig='phy_sig'
        file_end='.bdf.mrk'
        
        output_filename=self.files.out_filename(type_sig=type_sig,file_end=file_end,loc_folder='raw',short=True)
        self.stim_list=[self.stim_mrk.columns.tolist()]+self.stim_mrk.values.tolist()
        self.heps_list=[self.heps_df_mrk.columns.tolist()]+self.heps_df_mrk.values.tolist()
        self.ans_list=[self.ans_mrk.columns.tolist()]+self.ans_mrk.values.tolist()
        
        with open(output_filename,'w') as output:
            output.write('TL02\n')
        
            
            for line in self.stim_list[1:]:
                if line[2]=='normal':
                    output.write(f'{line[0]}\t{line[0]}\t"{line[30]}"\n')
                    output.write(f'{line[0]}\t{line[0]}\t"{line[31]}"\n')
                    output.write(f'{line[0]}\t{line[0]}\t"{line[32]}"\n')
            for line_hep in self.heps_list[1:]:
                if line_hep[2]=='normal':
                    output.write(f'{line_hep[0]}\t{line_hep[0]}\t"{line_hep[27]}"\n')
    
                    
            for line_ans in self.ans_list[1:]:
                if line_ans[3]=='normal':
                    output.write(f'{line_ans[1]}\t{line_ans[1]}\t"X{line_ans[31]}"\n')
                    output.write(f'{line_ans[1]}\t{line_ans[1]}\t"X{line_ans[32]}"\n')
                    output.write(f'{line_ans[1]}\t{line_ans[1]}\t"X{line_ans[33]}"\n')



    
    def get_annotations(self):
        

        
        start=[]
        
        description=[]
        
        for line in self.stim_list[1:]:
            start.append(line[0]/self.srate)
            description.append('vep/'+line[2]+'/'+line[3]+'/'+line[4]+'/'+line[5]+'/'+line[19])
        
        for line_hep in self.heps_list[1:]:
            start.append(line_hep[0]/self.srate)
            description.append('hep/'+line_hep[2]+'/'+line_hep[3]+'/'+line_hep[4]
                               +'/'+line_hep[15]+'/'+line_hep[16]+'/'+line_hep[32])
         
        
        for line_ans in self.ans_list[1:]:
            start.append(line_ans[1]/self.srate)
            description.append('xns/'+line_ans[3]+'/'+line_ans[4]+'/'+line_ans[5]+'/'+line_ans[6]+'/'+line_ans[20])
        

        
        
        
        len_zero= len(description)
        
        duration=np.zeros(len_zero)
        

        
        
        
        
        event_annot=mne.Annotations(start,duration,description) 
        
        return event_annot
        
    


#%%
