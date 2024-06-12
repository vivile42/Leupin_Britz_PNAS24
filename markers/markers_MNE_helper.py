#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 11:29:26 2021
functions to run to get mne structure and output
@author: leupinv
"""
import mne
import pandas as pd
import neurokit2 as nk
import markers.markers_constants as cs
import base.files_in_out as files_in_out
import feather
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel


class MarkersMNE:
    def __init__(self, files):

        self.files = files
        self.raw = mne.io.read_raw_bdf(
            self.files.current_file_dir, preload=True)
        try:
            self.raw.set_channel_types({'Erg1': 'resp', 'EXG1': 'ecg'})
        except:
            self.raw.set_channel_types({'Erg1': 'resp', 'EXG1-0': 'ecg'})
            self.raw.drop_channels(
                ['EXG1-1', 'EXG2', 'EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8'])
        self.montage = mne.channels.make_standard_montage('biosemi128')
        #self.montage=self.gen_montage()
        try:
            self.raw.set_montage(self.montage)
        except:
            self.raw.drop_channels('Erg2')
            self.raw.set_montage(self.montage)

        self.srate = self.raw.info['sfreq']
        self.eeg_exp = files.eeg_exp

    def merge_raws(self, raw_list):
        '''


        Parameters
        ----------
        raw_list : TYPE: list of raw mne files
            DESCRIPTION: Concatenate raw MNE file sto have single file for
            each subject and condition.

        Returns
        -------
        None.

        '''
        srate = []
        for raw in raw_list:
            srate.append(raw.info['sfreq'])

        if all(x == srate[0] for x in srate):

            self.raws = mne.concatenate_raws(raw_list)
        else:
            raise Exception('Careful this subject has different SR')

    def get_triggers(self):
        '''
        Gets events from stim channel and create first PD dataframe

        Returns
        -------
        None.

        '''
        self.events_coded = [
            [x[0], x[1], 256-(2**16-x[2])] for x in self.events]
        self.mrk = pd.DataFrame(self.events_coded)
        self.mrk.columns = ['TF', 'nul', 'trigger']
        self.mrk = self.mrk.drop(columns=["nul"])
        self.mrk.set_index('TF', inplace=True)

        if self.eeg_exp == 'tsk':

            df_2 = self.mrk.loc[(self.mrk['trigger'] == 1) | (self.mrk['trigger'] == 2) | (
                self.mrk['trigger'] == 10) | (self.mrk['trigger'] == 20)]
            df_3 = self.mrk.loc[(self.mrk['trigger'] == 3)
                                | (self.mrk['trigger'] == 4)]
            df_5 = self.mrk.loc[(self.mrk['trigger'] == 5)
                                | (self.mrk['trigger'] == 6)]
            self.mrk = pd.concat([df_2, df_3, df_5], axis=1)
            self.mrk.reset_index(inplace=True)
            self.mrk.columns = ['TF', 'trigger_stim',
                                'trigger_corr', 'trigger_aware']
        elif self.eeg_exp == 'int':
            self.df_mrk = self.mrk.loc[(self.mrk['trigger'] == 1) | (
                self.mrk['trigger'] == 2)]
            self.df_mrk['trigger'] = ['start' if x
                                      == 1 else 'end' for x in self.df_mrk['trigger']]
            if self.df_mrk['trigger'].tail(1).values == 'start':
                self.df_mrk = self.df_mrk[:-1]

            idx_df = self.df_mrk.copy()
            idx_df = idx_df[idx_df['trigger'] == 'start']
            idx_df.reset_index(inplace=True)
            idx_df.reset_index(inplace=True)
            merge = self.df_mrk.merge(
                idx_df, on=['TF', 'trigger'], how='outer')
            merge.fillna(method='pad', inplace=True)

            self.piv_int = merge.pivot(
                index='index', columns='trigger', values='TF')
            self.piv_int = self.piv_int[['start', 'end']]
            filename = self.files.out_filename(
                type_sig='raw', file_end='_triggers.csv', short=True, loc_folder='raw')
            self.piv_int.to_csv(filename)
        elif self.eeg_exp == 'flic':
            self.df_mrk = self.mrk.loc[(self.mrk['trigger'] == 1) | (
                self.mrk['trigger'] == 2)]
            self.df_mrk.drop(index=self.df_mrk.index[0], axis=0, inplace=True)
            self.df_mrk['trigger'] = ['C' if x
                                      == 1 else 'C' for x in self.df_mrk['trigger']]

    def get_rsp(self, raws, alert=False):
        '''
        Process rsp signal and returns rsp markers dataframe
        !!Important!! :  modification was done in rsp_peaks and amplitude_min was set to 0.22
        This parametres needs to be updated in any version of neurokit on which the code runs
        Otherwise different outputs may arise

        Returns
        -------
        None.

        '''
        if alert:
            raise ValueError(
                'remember to modify rsp peaks or double check and check alert off')
        self.rsp = raws.copy().pick_types(resp=True).get_data().flatten()*-1

        self.rsp_signals, self.rsp_info = nk.rsp_process(
            self.rsp, sampling_rate=self.srate)
        nk.rsp_plot(self.rsp_signals, sampling_rate=self.srate)
        fig_rsp =plt.gcf()
        # #Save fig output
        # output_filename=self.files.out_filename(type_sig=cs.type_sig_png,file_end='rsp_fig'+cs.file_end_png,short=True)
        # fig_rsp.savefig(output_filename,dpi=1000)
        self.report.add_figure(fig_rsp, title='rsp signal')

    def merge_rsp_DF(self):

        # Get dataframe to merge with stim mrk
        self.rsp_signals_pk = self.rsp_signals[['RSP_Peaks', 'RSP_Troughs']]
        self.rsp_signals_pk = self.rsp_signals_pk.loc[(
            self.rsp_signals_pk['RSP_Peaks'] == 1) | (self.rsp_signals_pk['RSP_Troughs'] == 1)]

        self.rsp_signals_pk['rsp_phase'] = [
            'exh' if x == 1 else 'inh' for x in self.rsp_signals_pk['RSP_Peaks']]

        Rsp_rate_df = self.rsp_signals_pk[self.rsp_signals_pk['rsp_phase'] == 'inh'].copy(
        )
        Rsp_rate_df.drop(['RSP_Peaks', 'RSP_Troughs'], axis=1, inplace=True)
        Rsp_rate_df.reset_index(inplace=True)
        Rsp_rate_df.rename(dict(index='TF'), axis=1, inplace=True)
        Rsp_rate_df['rsp_int'] = (
            Rsp_rate_df['TF'].diff().shift(-1))/self.srate
        Rsp_rate_df['RSP_Rate'] = 60/(Rsp_rate_df['rsp_int'])
        Rsp_rate_df['RSP_Rate_post'] = Rsp_rate_df['RSP_Rate'].shift(-1)
        # Add rolling window of 5 rsp rate preceding target
        Rsp_rate_df['RSP_Rate_rolling_5_before'] = Rsp_rate_df['RSP_Rate'].rolling(
            5, min_periods=1).mean()
        # Add rolling window of 5 rsp rate centered around target
        Rsp_rate_df['RSP_Rate_rolling_5_centered'] = Rsp_rate_df['RSP_Rate'].rolling(
            5, min_periods=1, center=True).mean()
        # Add rolling window of 5 rsp rate following target
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=2)
        Rsp_rate_df['RSP_Rate_rolling_5_after'] = Rsp_rate_df['rsp_int'].rolling(
            window=indexer, min_periods=1).mean()
        # Add rolling window of 10 heart rate variability before target
        Rsp_rate_df['RspRateVar_rolling_10_centered'] = Rsp_rate_df['rsp_int'].rolling(
            window=10, min_periods=2).std()
        # Add rolling window of 100 heart rate variability around  target
        Rsp_rate_df['RspRateVar_rolling_100_centered'] = Rsp_rate_df['RSP_Rate'].rolling(
            window=100, min_periods=2, center=True).std()

        self.rsp_df_mrk = self.rsp_signals_pk['rsp_phase'].copy().reset_index()
        self.rsp_df_mrk.columns = ['TF', 'rsp_phase']
        self.rsp_df_mrk = self.rsp_df_mrk.merge(
            self.rsp_signals['RSP_Rate'], 'inner', left_on='TF', right_index=True)
        self.rsp_df_mrk.rename(
            dict(RSP_Rate='RSP_Rate_precedent'), axis=1, inplace=True)
        #append rsp to signal to use later
        self.rsp_df_mrk = self.rsp_df_mrk.merge(Rsp_rate_df, 'left')

        self.rsp_df_mrk.fillna(method='pad', inplace=True)
        self.rsp_df_mrk.fillna(method='backfill', inplace=True)
        #Save dataframe + image
        # out_filename=self.files.out_filename(type_sig=cs.type_sig_physig,file_end='rsp_sig'+cs.file_end_feather,short=True)
        # feather.write_dataframe(self.rsp_signals, out_filename)

    def correct_resp(self):
        self.plot_rsp(correction=True)

    def get_card(self, raws, cut_idx=None):
        '''
        process ecg signals without correcting for outliers peaks, focuses on
        getting the peak at the right moment. For computing HRV parametres signals
        is then cleaned

        Returns
        -------
        None.

        '''
        method = 'engzeemod2012'
        #method='neurokit'
        self.ecg_sig = raws.copy().pick_types(ecg=True).get_data().flatten()
        print(len(self.ecg_sig))

        print(len(self.ecg_sig))
        ecg_signal = nk.signal_sanitize(self.ecg_sig)

        ecg_cleaned = nk.ecg_clean(
            ecg_signal, sampling_rate=self.srate, method='neurokit')
        # R-peaks
        instant_peaks, rpeaks, = nk.ecg_peaks(
            ecg_cleaned=ecg_cleaned, sampling_rate=self.srate, method=method, correct_artifacts=False
        )

        rate = nk.signal_rate(rpeaks, sampling_rate=self.srate,
                              desired_length=len(ecg_cleaned))

        quality = nk.ecg_quality(
            ecg_cleaned, rpeaks=None, sampling_rate=self.srate)

        signals = pd.DataFrame(
            {"ECG_Raw": ecg_signal, "ECG_Clean": ecg_cleaned, "ECG_Rate": rate, "ECG_Quality": quality})

        # Additional info of the ecg signal
        delineate_signal, delineate_info = nk.ecg_delineate(
            ecg_cleaned=ecg_cleaned, rpeaks=rpeaks, sampling_rate=self.srate
        )

        cardiac_phase = nk.ecg_phase(
            ecg_cleaned=ecg_cleaned, rpeaks=rpeaks, delineate_info=delineate_info)

        self.ecg_signals = pd.concat(
            [signals, instant_peaks, delineate_signal, cardiac_phase], axis=1)

        # Save dataframe + image
        #image
        nk.ecg_plot(self.ecg_signals, sampling_rate=self.srate)
        # output_filename=self.files.out_filename(type_sig=cs.type_sig_png,file_end='ecg_fig'+cs.file_end_png,short=True)
        # ecg_fig.savefig(output_filename,dpi=1000)
        ecg_fig=plt.gcf()
        self.report.add_figure(ecg_fig, title='R peaks plot')

        # DF
        out_filename = self.files.out_filename(
            type_sig=cs.type_sig_physig, file_end='ecg_sig'+cs.file_end_feather, short=True)
        feather.write_dataframe(self.ecg_signals, out_filename)

        self.get_ecg_stim_DF()
        self.get_ecg_hep_DF()

    # def improve_Card(self):
        #This is only for twitching and testing
    #     self.ecg_sig=self.raws.copy().pick_types(ecg=True).get_data().flatten()
    #     ecg_signal = nk.signal_sanitize(self.ecg_sig)
    #     method='engzeemod2012'

    #     ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=self.srate)
    #     # R-peaks
    #     instant_peaks, rpeaks, = nk.ecg_peaks(
    #         ecg_cleaned=ecg_cleaned, sampling_rate=self.srate, method=method, correct_artifacts=False
    #     )

    #     plt.figure()
    #     plt.plot(ecg_cleaned)
    #     print(rpeaks)
    #     print(instant_peaks)
    #     ECG_peaks=instant_peaks['ECG_R_Peaks']
    #     ECG_peaks=ECG_peaks[ECG_peaks==1]

    #     plt.plot(ECG_peaks.index,ecg_cleaned[ECG_peaks.index],'o')
    #     plt.title(method)

    #     return instant_peaks['ECG_R_Peaks'], ecg_cleaned

    def get_ecg_stim_DF(self):
        '''
        Get dataframe to merge with stim mrk

        Returns
        -------
        None.

        '''
        ecg_signals_pk = self.ecg_signals[['ECG_R_Peaks', 'ECG_T_Offsets']]
        ecg_signals_pk = ecg_signals_pk.loc[(ecg_signals_pk['ECG_R_Peaks'] == 1) | (
            ecg_signals_pk['ECG_T_Offsets'] == 1)]

        ecg_signals_pk['cardiac_phase'] = ['R' if x
                                           == 1 else 'T' for x in ecg_signals_pk['ECG_R_Peaks']]
        ecg_signals_pk.reset_index(inplace=True)
        ecg_signals_pk.rename(dict(index='TF'), axis=1, inplace=True)
        R_df = self.get_HeartRate(ecg_signals_pk)
        self.cardiac_mrk_stim = ecg_signals_pk.merge(R_df, 'left')
        self.cardiac_mrk_stim.fillna(method='pad', inplace=True)

    def get_HeartRate(self, ecg_sig):
        R_df = ecg_sig.copy()

        # Add crazy stuff based on RR interval
        R_df = R_df[R_df['cardiac_phase'] == 'R']

        R_df['RRI'] = (R_df['TF'].diff().shift(-1))/self.srate
        R_df['HeartRate'] = 60 / (R_df['RRI'])
        R_df['HeartRate_post'] = R_df['HeartRate'].shift(-1)
        # Add rolling window of 5 heart rate preceding target
        R_df['HeartRate_rolling_5_before'] = R_df['HeartRate'].rolling(
            5, min_periods=1).mean()
        # Add rolling window of 5 heart rate centered around target
        R_df['HeartRate_rolling_5_centered'] = R_df['HeartRate'].rolling(
            5, min_periods=1, center=True).mean()
        # Add rolling window of 5 heart rate following target
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=2)
        R_df['HeartRate_rolling_5_after'] = R_df['HeartRate'].rolling(
            window=indexer, min_periods=1).mean()

        # Add rolling window of 10 heart rate variability before target
        R_df['HeartRateVar_rolling_10_centered'] = R_df['RRI'].rolling(
            window=10, min_periods=2).std()
        # Add rolling window of 100 heart rate variability around  target
        R_df['HeartRateVar_rolling_100_centered'] = R_df['RRI'].rolling(
            window=100, min_periods=2, center=True).std()
        #ecg_signals_pk.drop('RRI',axis=1,inplace=True)
        R_df = R_df.merge(
            self.ecg_signals['ECG_Rate'], 'inner', left_on='TF', right_index=True)
        R_df.rename(dict(ECG_Rate='HeartRate_precedent'), axis=1, inplace=True)
        return R_df

    def get_ecg_hep_DF(self):
        '''
         Get dataframe to merge with stim mrk
         need to create DF with t-wave peak instead of offset

        Returns
        -------
        None.

        '''

        ecg_signals_Tpk = self.ecg_signals[['ECG_R_Peaks', 'ECG_T_Peaks']]
        ecg_signals_Tpk = ecg_signals_Tpk.loc[(ecg_signals_Tpk['ECG_R_Peaks'] == 1) | (
            ecg_signals_Tpk['ECG_T_Peaks'] == 1)]

        ecg_signals_Tpk['cardiac_phase'] = ['R' if x
                                            == 1 else 'T' for x in ecg_signals_Tpk['ECG_R_Peaks']]

        ecg_signals_Tpk.reset_index(inplace=True)
        ecg_signals_Tpk.rename(dict(index='TF'), axis=1, inplace=True)

        self.cardiac_mrk_Tpk = ecg_signals_Tpk.copy()
        R_df = self.get_HeartRate(ecg_signals_Tpk)
        self.cardiac_mrk_Tpk = self.cardiac_mrk_Tpk.merge(R_df, 'left')
        self.cardiac_mrk_Tpk.fillna(method='pad', inplace=True)

    def get_ds_eeg(self, raw, open_file=False, events_from_ds=False):
        '''

        Parameters
        ----------
        open_file : TYPE, Boleen
            DESCRIPTION. The default is False. If False it will downsample the
            Data, if True then it will open the already downsampled file
            Also starts get triggers and events
        events_from_ds: TYPE, Boleen
            DESCRIPTION. The default is false. Switch to focus next preprocessing
            steps on the DS. E.g. computes cardiac and triggers based on the DS

        Returns
        -------
        None.

        '''

        if not open_file:
            self.eeg_filt = raw.copy().filter(cs.l_filt, cs.h_filt, method='fir', n_jobs=-1)
            self.eeg_ds = self.eeg_filt.resample(sfreq=cs.sfreq, n_jobs=-1)

            # fig=plt.gcf()
            # output_filename=self.files.out_filename(type_sig=cs.type_sig_png,file_end='psd_fig'+cs.file_end_png,short=True)
            # fig.savefig(output_filename,dpi=1000)
            #self.srate=self.eeg_ds.info['sfreq']
            #self.raws=self.eeg_ds

            self.eeg_ds.set_eeg_reference('average', projection=True)

        else:
            get_fif = files_in_out.GetFiles(
                filepath='raw', condition=self.files.condition, g_num=self.files.g_num, eeg_format='.fif')

            fif_taskfiles = get_fif.condition_files[0]

            self.eeg_ds = mne.io.read_raw_fif(fif_taskfiles, preload=True)



        if events_from_ds:
            self.events = mne.find_events(
                self.eeg_ds.copy(), consecutive=True, shortest_event=1)
            #set SR to DS one
            self.srate = cs.sfreq
        else:
            self.events = mne.find_events(
                raw.copy(), consecutive=True, shortest_event=1)
        #self.get_triggers()
        self.save_df()

        self.report = files_in_out.init_report()
        self.report.add_raw(self.eeg_ds,'raw downsampled',replace=True)

    def update_annot(self, annot, append=False):
        '''


        Parameters
        ----------
        annot : TYPE Annotation class from MNE
            DESCRIPTION. update annotations based on artefact rejection


        Returns
        -------
        None.

        '''
        if not append:
            self.eeg_ds.set_annotations(annot)
        elif append:
            onset = annot.onset
            duration = annot.duration
            label = annot.description
            self.eeg_ds.annotations.append(onset, duration, label)

    def save_df(self, short=True):
        '''
        Save downsampled dataset,


        '''

        output_filename = self.files.out_filename(
            type_sig='phy_sig', file_end='_ds_eeg-raw.fif', loc_folder='raw', short=short)

        self.eeg_ds.save(output_filename, overwrite=True)

    def get_bad_interval(self, resp=False):
        '''


        Parameters
        ----------
        resp : TYPE boolean
            DESCRIPTION. The default is False. if True returns bads for breathing
            and updates that period in annotations

        Returns
        -------
        None.

        '''
        annotations = self.eeg_ds.annotations
        onset = annotations.onset
        onset_diff = np.diff(onset)

        self.onset_bad = [x+cs.buff_int for x,
                          y in zip(onset, onset_diff) if y > cs.int_min]

        self.duration_bad = [y-cs.buff_int for x,
                             y in zip(onset, onset_diff) if y > cs.int_min]

        self.description_bad = ['BAD_interval']*len(self.onset_bad)

        #self.annotations_bad=mne.Annotations(self.onset_bad, self.duration_bad, self.description_bad)

        if resp:
            self.get_rsp_Bads()
            self.eeg_ds.annotations.append(
                self.onset_bad, self.duration_bad, self.description_bad)

    def remove_bad_sig(self, sig, resp=False):
        '''


        Parameters
        ----------
        sig : TYPE pandas dataframe, signal to clean
            DESCRIPTION.
        resp : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        clean_sig : TYPE
            DESCRIPTION.

        '''
        self.get_bad_interval(resp=resp)

        index_list = [(round(x*self.srate), round((x+y)*self.srate))
                      for x, y in zip(self.onset_bad, self.duration_bad)]

        values_arr = [np.arange(x[0], x[1]) for x in index_list]
        idx_list = np.concatenate(values_arr)

        clean_sig = np.delete(sig, idx_list)

        return clean_sig

    def get_rsp_Bads(self):

        #Find moments in which RSP rate is outliers (this helps us identify artefactual periods)
        rsp_std_plus = cs.up_std * \
            self.rsp_signals['RSP_Rate'].std(
            )+self.rsp_signals['RSP_Rate'].mean()
        rsp_std_minus = self.rsp_signals['RSP_Rate'].mean(
            )-cs.low_std*self.rsp_signals['RSP_Rate'].std()
        self.rsp_signals['RSP_Bad'] = [1 if x > rsp_std_plus or x
                                       < rsp_std_minus else 0 for x in self.rsp_signals['RSP_Rate']]
        self.mask = self.rsp_signals['RSP_Bad'].to_numpy()

        if self.mask[-1] == 1:
            self.mask[-1] = 0

        mask_edge = np.diff(self.mask)

        self.plot_rsp(mask=True)

        start = np.where(mask_edge == 1)

        ends = np.where(mask_edge == -1)

        duration = (ends[0]-start[0])/self.srate
        label = ['BAD_rsp']

        self.eeg_ds.annotations.append(
                start[0]/self.srate, duration, label)
        # annot=mne.Annotations(start[0]/self.srate, duration ,label)
        # self.eeg_ds.set_annotations(annot)
        #self.eeg_ds.plot()

    def plot_rsp(self, mask=False):
        plt.figure()
        rsp_signals = self.rsp_signals
        rsp_signals.reset_index(inplace=True)
        plt.plot(self.rsp_signals['RSP_Rate'])
        plt.plot((self.rsp_signals['RSP_Clean'])*10000)
        plt.plot(rsp_signals['index'].loc[rsp_signals['RSP_Peaks'] == 1],
                 (rsp_signals['RSP_Clean'].loc[rsp_signals['RSP_Peaks'] == 1])*10000, 'o')
        plt.plot(rsp_signals['index'].loc[rsp_signals['RSP_Troughs'] == 1], (
            rsp_signals['RSP_Clean'].loc[rsp_signals['RSP_Troughs'] == 1])*10000, 'o')
        if mask != False:

            plt.plot(self.mask*100)

        plt.title(f'{self.files.g_num}')

        rsp_fig = plt.gcf()

        self.report.add_figure(rsp_fig, title='Resp mask')

    def get_HRV(self):
        '''
        Generate HRV parametres, ecg signals here obtained are cleaned to get more
        precise HRV and HR

        Returns
        -------
        None.

        '''
        ecg_sig_sum = self.remove_bad_sig(self.ecg_sig)
        self.ecg_signals_sum, self.info_hrv = nk.ecg_process(
            ecg_sig_sum, sampling_rate=self.srate)

        HRV = nk.hrv(self.info_hrv, sampling_rate=self.srate, show=True)
        HRV_fig = plt.gcf()
        r_peaks_hrv = self.ecg_signals_sum[self.ecg_signals_sum['ECG_R_Peaks'] == 1]
        r_peaks_hrv.reset_index(inplace=True)
        r_peaks_hrv = r_peaks_hrv['index']

        # HRV_lomb=nk.hrv_frequency(r_peaks_hrv,sampling_rate=self.srate,show=True,psd_method='lomb')

        type_sig = 'phy_sig'
        file_end = 'HRV.feather'

        output_filename = self.files.out_filename(
            type_sig=type_sig, file_end=file_end, short=True)
        feather.write_dataframe(HRV, output_filename)

        # file_end='HRV_lomb.feather'

        # output_filename=self.files.out_filename(type_sig=type_sig,file_end=file_end,short=True)
        # feather.write_dataframe(HRV_lomb, output_filename)

        # type_sig='png'
        # file_end='hrv_fig.png'

        # output_filename=self.files.out_filename(type_sig=type_sig,file_end=file_end,short=True)
        # HRV_fig.savefig(output_filename,dpi=2000)

        self.report.add_figure(HRV_fig, title='HRV results')

    def get_RSA(self):
        '''
        Generates RSA and RRV

        Returns
        -------
        rsa : TYPE neurokit output in pandas DF
            DESCRIPTION. RSA parametres
        rrv : TYPE neurokit output in pandas DF
            DESCRIPTION. rrv parametres

        '''
        resp = self.remove_bad_sig(self.rsp, resp=True)
        self.rsp_signals_sum, rsp_info = nk.rsp_process(
            resp, sampling_rate=self.srate)
        rsa = nk.hrv_rsa(self.ecg_signals_sum, rsp_signals=self.rsp_signals_sum,
                         rpeaks=self.info_hrv, sampling_rate=self.srate, continuous=False)
        rsa_df = pd.DataFrame.from_dict(rsa, orient='index').T

        type_sig = 'phy_sig'
        file_end = 'RSA.feather'

        output_filename = self.files.out_filename(
            type_sig=type_sig, file_end=file_end, short=True)

        feather.write_dataframe(rsa_df, output_filename)

        #fig_rsp=nk.rsp_plot(self.rsp_signals_sum,sampling_rate=self.srate)

        rsp_rate = nk.signal_rate(self.rsp_signals_sum)
        rsp_peaks = nk.rsp_peaks(self.rsp_signals_sum)


        rrv = nk.rsp_rrv(rsp_rate=rsp_rate, troughs=rsp_peaks[1]['RSP_Troughs'],
                         sampling_rate=self.srate, show=True, silent=False)

        rrv_fig = plt.gcf()
        # type_sig='png'
        # file_end='rrv_fig.png'

        # output_filename=self.files.out_filename(type_sig=type_sig,file_end=file_end,short=True)
        # rrv_fig.savefig(output_filename,dpi=2000)

        self.report.add_figure(rrv_fig, title='RRV')

        type_sig = 'phy_sig'
        file_end = 'rrv.feather'

        output_filename = self.files.out_filename(
            type_sig=type_sig, file_end=file_end, short=True)

        feather.write_dataframe(rrv, output_filename)

        return rsa, rrv
        return rsa, rrv

#%%
