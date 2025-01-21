# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 09:59:34 2021

@author: Engi
"""



import beavioral.behav_constants as cs
import beavioral.behav_helper as hp

import base.files_in_out as files_in_out 
import base.base_constants as b_cs
import numpy as np

import pandas as pd
import feather

# before rerunning make sure that df is filtered to get only
# signal type = vep
df_list=[]

for g_n in b_cs.G_N:
    for cond in cs.condition[0]:
        files = files_in_out.GetFiles(filepath=cs.datafolder,eeg_format='off',condition=cond, g_num=g_n)
        files.filter_file(filters='metadata_filt_rsp.feather')
        df=hp.Behav_DF(files)

        df.find_thresh()
        df.compute_distribution()


        
        df_list.append(df.df)
    
def_df=pd.concat(df_list)
param_acc=pd.pivot_table(def_df,values=['corr','CACU_corr'],index=['g_num'])
filename=('behav_acc_n.feather')

fileout=cs.out+filename

feather.write_dataframe(param_acc, fileout)


        
