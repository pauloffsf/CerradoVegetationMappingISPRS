# Mapping the Brazilian Savanna's Natural Vegetation: A SAR-Optical Uncertainty-Aware Deep Learning Approach
This repository contains materials for the paper accepted on the ISPRS Journal of Photogrammetry and Remote Sensing

## Dataset:
The dataset can be downloaded at: https://www.doi.org/10.17026/dans-xne-cdhn

## Code:

The code here presented consists of several different files to perform the Training, Testing and analysis of the methodology presented in the paper.

### Needed Packages:
1. Tensorflow (< Version 2.13. Newer versions of Tensorflow will require changes in the Baseline.py script)
2. Tifffile
3. OpenCV
4. Numpy
5. CVNN (https://github.com/NEGU93/cvnn)
6. tqdm
7. pandas
8. GDAL
9. matplotlib
10. sklearn
11. pickle
12. tensorflow_probability
13. PIL

### For training:
First adjust the UnetParams.py file accordingly:
1. Addresses for the Training and Validation data
2. Address to save the checkpoint files and address to save the test results (in the training this will be empty)
3. BATCH_SZ according to the GPU memory available
4. NUM_CATEGORIES (4 if level-1 training, or 6 if level-2 training)
5. NUM_EPOCHS
6. DATA_USAGE
7. NETWORK

Run "python NetworkRun.py train"

### For testing:
Check the Scripts GeneralizationTestStill.py for the deterministic test, or GeneralizationTestDrop.py for MCDropOut test, by running the function GenTest()

### For Calibrating uncertainties:
For now, only one model in seg_models is prepaired to be calibrated (labelsuperFPB_early2_mod). However, it is easy to implement other methods by following the example of the implemented model. Check seg_model/UnetFull.py and function uncertaintyTemperatureCalibration to implement the possibility of Temperature calibration of other models.

To calibrate the labelsuperFPB_early2_mod, change on UnetParams.py accordingly:
1. NETWORK
2. OGRMODEL2CALIB
3. OGRMODEL2CALIBLOG

then run "python NetworkRun.py train"

after calibrated model, run GeneralizationTestDrop.py, maintaining the parameters.

### Hierarchical combination
This is performed by the functions in MatchLv1Lv2.py. It is recomended to combine the uncertainties just after the calibration.

### Uncertainty evaluation
The Graphs for the uncertainty evaluation are obtained by the functions described in Uncertainty_Graph.py
After analysing the Graph and determining the Threshold, we can use the functions in UncertaintyMasks.py to obtain the Incorrect-Certain and Correct-Uncertain masks.

### Accuracy Assessment
The accuracy assessment according to the Reference Map is performed directly when doing the Generalization tests, by the functions in MetricsCalc.py
For the point-wise accuracy assessment, the functions are presented in PointTestingMapBiomas.py (for the MapBiomas points) and PointTesting.py for the Bendini et. al. (2020) field work points. With the resulting confusion matrixes, the user must apply the Olofsson, et. al. (2014) protocol (code for the protocol was not produced and the results for the article were produced on a excel sheet).

