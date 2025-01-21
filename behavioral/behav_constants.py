# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 09:59:09 2021

@author: Engi
"""


import os
import platform



platform.system()

# define starting datafolder 
import sys
if platform.system()=='Darwin':
    sys.path.append('/Users/leupinv/BBC/WP1/data/Code/python/BBC')
    os.chdir('/Volumes/BBC/BBC/WP1/data/EEG/tsk/')
    sys.path.append('/Users/leupinv/BBC/WP1/data/Code/python/BBC')



elif platform.system()=='Windows':
    os.chdir('Z:/BBC/WP1/data/EEG/tsk')

    sys.path.append('C:/Users/Vivi/switchdrive/BBC/WP1/data/Code/python/BBC')
    #os.chdir('c:/Users/Engi/all/BBC/WP1/data/EEG/tsk')
elif platform.system()=='Linux':
    os.chdir('/run/user/1000/gvfs/smb-share:server=bigdata,share=arts/Psycho/BBC/BBC/WP1/data/EEG/tsk')
    #os.chdir('c:/Users/Engi/all/BBC/WP1/data/EEG/tsk')
    
datafolder='preproc'


condition=['n']

out='ana/behavioral/'