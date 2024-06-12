Code accompaning Leupin and Britz (2024) 
This code includes preprocessing and analyses of EEG data available on request.
# Code organization
The base folder contains some helper function to filter through the directories. Each folder is generally organized with a main, helper and constants script.
- The main scripts contain the code that must be run.
- The helper scripts contain the helper functions and classees used to run the code.
- The constant files contain constants that are called in the script.


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
1) Marker: generates markers and analyses cardiac and respiratory signals. Then classifies each stimulus into the physiological phase.
2) Epochs: generates epochs before cleaning and computes ICA solutions.
6) ICA: jupyter notebook to be applyied to each subject to manually select ICA.
7) Autoreject: cleans the epoched data after the ICA.
8) Evoked_MNE: computes evoked potentials for each condition and each subject.
9) Source compute_fwd_solution: computes the fwd solution (requires manual coregistration).
10) Source_main: applyies inverse model to ERPs to compute sources.
## Analyses
Analyses are contained in the stats folder.
### Behavioral
1) Behav_rev.ipynb contains figures and descriptive statistics.
2) Behavioral_GLM.Rmd contains general linear models.
### Sensor space
1) Sensor_space_stats.ipynb contains all analysis and figure output for the ERP analyses.
2) Stats_helper contains helper functions used in sensor_space_stats.ipynb.
### Source space
1) Source_statistics.ipynb contains all analysis and figure output for the source space analyses.
2) Inverse_stats_helper.py contains helper functions used in source_statistics.ipynb.





