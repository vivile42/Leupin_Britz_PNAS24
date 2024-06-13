Code accompanying Leupin and Britz, PNAS (2024)
We here provide the code for preprocessing, epoching, and analysis of EEG data in sensor and source space. EEG data can be made available upon request.
# Code organization
Each folder is generally organized with a main, helper and constants script.
•	The main scripts contain the code that must be run.
•	The helper scripts contain the helper functions and classes used to run the code.
•	The constant files contain constants that are called in the script.
The base folder contains some helper functions to filter through the data directories.

## Preprocessing
### Order to run preprocessing
1) markers/markers_main.py 
2) epochs/epochs_main.py
3) ICA
4) evoked/autoreject_main.py
5) evoked/evoked_MNE_main.py
6) source/compute_fwd_solution.ipynb
7) source/source_main.py
### Description
1)	markers: analyzes cardiac and respiratory signals and generates markers to classify each stimulus according to the behavioral response and the cardiac / respiratory phase.
2)	epochs: segments the EEG into epochs before artifact rejection and computes ICA solutions.
3)	ICA: jupyter notebook to be applied to each subject to manually select ICA components to be rejected.
4)	evoked:
  - autoreject: used the Autoreject procedure to clean the epoched data after the ICA.
  - evoked_MNE: computes evoked potentials for each condition and each subject.
5)	source:
  - compute_fwd_solution: jupyter notebook to compute the forward solution (requires manual coregistration of electrodes).
  - source_main: applies inverse solution of the forward model to ERPs to compute sources for each condition and each subject.

## Analyses
The stats folder contains the code for statistical analyses.
### Behavioral
1)	Behavioral_GLM.Rmd: R markdown script to run the General Linear Mixed Models.
2)	Behav_rev.ipynb: jupyter notebook to compute descriptive statistics and code to generate figures.
### Sensor space
1)	Sensor_space_stats.ipynb: jupyter notebook that contains all analysis and figure output for the ERP analyses.
2)	Stats_helper contains helper functions used in sensor_space_stats.ipynb.
### Source space
1)	Source_statistics.ipynb: jupyter notebook that contains all analysis and figure output for the source space analyses.
2)	Inverse_stats_helper.py contains helper functions used in source_statistics.ipynb.






