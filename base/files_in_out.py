#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 09:37:11 2021

Inport and export data handler

@author: leupinv
Helper functions for working with directories and subdirectories
"""
#%% Scan directories and sub directories

import os
import mne
import time

def filter_list(list_,value):
    filter_list=[x for x in list_ if value in x]
    return filter_list

# def getListOfFiles(dirName,g_num='g'):
#     start_time=time.time()
#     # create a list of file and sub directories
#     # names in the given directory
#     allFiles = list()
#     # Iterate over all the entries
#     for root,dirs,files in os.walk(dirName):
#             #fullPath = os.path.join(dirName, entry)
#             # If entry is a directory then get the list of files in this directory
#                 for file in files:
#                     allFiles.append(os.path.join(root, file))
                   
                    
#     print("time:", time.time()- start_time)
#     return allFiles
#         # else:
#         #     if g_num in entry:
#         #     # Create full path
#         #         fullPath = os.path.join(dirName, entry)
#         #         # If entry is a directory then get the list of files in this directory
#         #         if entry.isdir(follow_symlinks=False):
#         #             yield from getListOfFiles(entry.path)
#         #         else:
#         #             yield entry


def getListOfFiles(dirName,g_num='g'):
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        if g_num==None:
            fullPath = os.path.join(dirName, entry)
            # If entry is a directory then get the list of files in this directory
            if os.path.isdir(fullPath):
                allFiles = allFiles + getListOfFiles(fullPath,g_num)
            else:
                allFiles.append(fullPath)

        else:
            if g_num in entry:
            # Create full path
                fullPath = os.path.join(dirName, entry)
                # If entry is a directory then get the list of files in this directory
                if os.path.isdir(fullPath):
                    allFiles = allFiles + getListOfFiles(fullPath,g_num)
                else:
                    allFiles.append(fullPath)

    return allFiles



    
def init_report():
    report = mne.Report(verbose=True,raw_psd=True)
    return report

def open_report(g_num,condition):
    file_end='report.h5'

    files=GetFiles(filepath='preproc',g_num=g_num,eeg_format=file_end,condition=condition)
    filename=files.condition_files[0]
    print(filename)
    report = mne.open_report(filename)

    
    return report

def save_report(files,report,final=False,short=False):
    type_sig='report'
    if final:
        file_end='report.html'
    else:    
        file_end='report.h5'

    filename=files.out_filename(type_sig=type_sig,file_end=file_end,short=short)
    report.save(filename, overwrite=True)
    

class GetFiles:
    def __init__(self,filepath,condition=None,g_num='g',eeg_format='bdf',eeg_exp='tsk'):
        """Default g_num=g,eeg_format='bdf',eeg_exp='tsk'"""
        self.filepath = filepath
        self.g_num = g_num
        self.fflist=getListOfFiles(self.filepath,self.g_num)
        self.eeg_format=eeg_format
        self.eeg_exp=eeg_exp
        self.find_files()
        self.condition=condition
        if self.condition!= None:
            self.select_condition(self.condition)



    def find_files(self):
        self.taskfiles=[x for x in self.fflist if self.eeg_exp in x and x.endswith(self.eeg_format) ]

    def select_condition(self,condition):
        self.condition=condition
        if self.eeg_format=='off':
            self.condition_files=[x for x in self.fflist if '_'+condition+'_' in x]
        else:     
            self.condition_files=[x for x in self.taskfiles if '_'+condition+'_' in x]

        self.condition_nfiles=len(self.condition_files)

    def get_info(self,index=0,end_fix=-4,start_fix=4,short_fix=2):
        """This can be easy called in a loop following find files.. e.g. for i in range len(taskfiles)"""
        if self.condition_files!=None:
              self.current_file_dir=self.condition_files[index]
        else:
            self.current_file_dir=self.taskfiles[index]

        self.fixed_len=len(self.filepath)+start_fix
        self.current_filename= self.current_file_dir[self.fixed_len:end_fix]
        self.short_name=self.current_file_dir[self.fixed_len:end_fix-short_fix]

    def get_names(self,index=0):
        if self.condition_files!=None:
            self.current_file_dir=self.condition_files[index]
        else:
            self.current_file_dir=self.taskfiles[index]

        self.current_filename=self.current_file_dir.split('/')[-1]


        self.short_name=self.current_filename.split('.')[0]
        print(self.short_name)
    def filter_file(self,filters):
        self.filt=[ x for x in self.condition_files if filters in x]
        self.current_file_dir= self.filt[0]
        





    def out_filename(self,type_sig,file_end,loc_folder='preproc',short=False):
        """loc_folder: indicate if preproc(default) or raw, type sig:subfolder inside preproc, file_end=specific file name"""

        if loc_folder=='preproc':
            directory=loc_folder+'/'+self.g_num+'/'+self.g_num+'_'+type_sig
            if short:
                output_filename=directory+'/'+self.short_name+'_'+file_end
            else:
                output_filename=directory+'/'+self.current_filename+'_'+file_end
        else:
            directory=loc_folder+'/'+self.g_num
            if short:
                output_filename=directory+'/'+self.short_name+file_end

            else:
                output_filename=directory+'/'+self.current_filename+file_end
        if not os.path.exists(directory):
            os.makedirs(directory)

        return output_filename
