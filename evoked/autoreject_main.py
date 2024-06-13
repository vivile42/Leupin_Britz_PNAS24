# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 15:20:18 2021
run autoreject on epoched data
output cleaned epoched data
@author: Viviana Leupin
"""

import evoked.autoreject_helper as auto_hp
from base.files_in_out import GetFiles,filter_list,getListOfFiles
import base.files_in_out as in_out
import evoked.evoked_constants as cs
import base.base_constants as b_cs



# In[main]




for g_n in b_cs.G_N:
    for cond in cs.condition:
        files = GetFiles(filepath=cs.datafolder,
                                      condition=cond,g_num=g_n,
                                      eeg_format=cs.eeg_format)

        files.condition_files=filter_list(files.condition_files,'_rec_')


        for n,file in enumerate(files.condition_files):
            
            files.get_info(index=n,end_fix=-12,start_fix=16,short_fix=0)
            filename=files.current_file_dir
            
            auto_rej=auto_hp.AutoHelp(file)
            auto_rej.report=in_out.open_report(g_n,cond)
            auto_rej.get_epochs()
            auto_rej.compute_autorej()
            auto_rej.save_output(files)
            in_out.save_report(files, auto_rej.report,final=True)

        
