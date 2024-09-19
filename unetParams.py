import os

#Data IO Params
#Addresses for the training input
TRAIN_S1_DIR = './Dataset/train/Data/'
TRAIN_S2_DIR = TRAIN_S1_DIR
TRAIN_LABEL_DIR = './Dataset/train/GT_L2/'
TRAIN_LABEL30_DIR = './Dataset/train/GT_L2/'
TRAIN_LABEL_NEW = './Dataset/train/NewGT/'

#if not os.path.isdir(TRAIN_LABEL_NEW):
#    os.makedirs(TRAIN_LABEL_NEW)

#Addresses for the validation input
VALID_S1_DIR = './Dataset/valid/Data/'
VALID_S2_DIR = VALID_S1_DIR
VALID_LABEL_DIR = './Dataset/valid/GT_L2/'
VALID_LABEL30_DIR = './Dataset/valid/GT_L2/'
VALID_LABEL_NEW = './Dataset/valid/NewGT/'

#if not os.path.isdir(VALID_LABEL_NEW):
#    os.makedirs(VALID_LABEL_NEW)

#Address where the weights will be saved.
CHECKPOINT_DIR = './Checkpoint/'


#Addresses where the Test images are and where the inferences for the generalization will be saved.
ORG_TEST_DIR = './GenTest/'
TEST_DIR = './GenTest/test/'
OUTPUT_DIR = './GenTest/patches/'
ASSEMBLE_OUT_DIR = './GenTest/Output/'
TEST_MODEL_DIR = './Checkpoint/' #weights.04.hdf5 #para o focal loss.


#When Calibration (address where the weights of the model to be calibrated is)
OGRMODEL2CALIB = './Checkpoint/'
OGRMODEL2CALIBLOG = os.path.join(OGRMODEL2CALIB, 'log.csv')


if not os.path.isdir(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

if not os.path.isdir(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

if not os.path.isdir(ASSEMBLE_OUT_DIR):
    os.makedirs(ASSEMBLE_OUT_DIR)


#If you want to continue trainning from previously trained weights.
CONTINUE_TRAINING = False
CONTINUE_MODEL_FILE_DIR = TEST_MODEL_DIR
SAVE_CHECK_EXT = '.hdf5'
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, 'weights.{epoch:02d}'+SAVE_CHECK_EXT)

#file where the training information will be saved
LOGFILE = './Checkpoint/log.csv'


#definition of certain labels to define each type of data
LABEL_FILE_STR = 'GT'
LABEL30_FILE_STR = 'MN'
LABEL_FILE_EXT = 'tif'
S1_FILE_STR = 'S1'
S1_FILE_EXT = 'tif'
S2_FILE_STR = 'S2'
S2_FILE_EXT = 'tif'
LABEL_PREC_STR = 'PREC'
LABEL_STD_STR = 'STD'
LABEL_MCLU_STR = 'MCLU'
LABEL_ENT_STR = 'ENT'
LABEL_DELPE_STR = 'DELT'
CLSPRED_FILE_STR = 'CLS'
CLSPRED_FILE_STR_30 = 'CL3'

#Model Training and Testing Params
GPUS = '0'  # GPU indices to restrict usage
NUM_CHANNELS_S1 = 7
NUM_CHANNELS_S2 = 10 =
S1_IN_DB = False
BATCH_SZ = 5
IMG_SZ = (384, 384)  # this code assumes all images in the training set have the same numbers of rows and columns
IGNORE_VALUE = -10000  # nan is set to this for ignore purposes later
NUM_CATEGORIES = 4  # use valu 4 for level 1 or 6 for level 2:  4 = 1 - Antropic / #2 - Water / #3 - Natural. / #0-Uknown class or trash or padding // 6 = 1-non vegetated / 2 - Grassland / 3 - Forest / 4 - Savanna / 5 - Secondary vegetation / 0 - all others.
MODEL_SAVE_PERIOD = 1  # how many epochs between model checkpoint saves
NUM_EPOCHS = 50  # total number of epochs to train with
BINARY_CONF_TH = 0.4 #not used, since the model is not deciding on binary cathegories.
NUM_DROPOUT_ITER = 10
OPTIMIZER = 'Adam' #porque adam? faz sentido? ou melhor modificar?

ENCODER_WEIGHTS = None

DATA_USAGE = 'S1I_S2_Sep' #Options: S1_I_only, S1_only, S2_only, S1S2All, S1_S2_Sep, S2S1IAll, S1I_S2_Sep, S1IV_S2_Sep, S2F_only, S1F_only, S1IF_only, S1IS2FAll, S1S2FAll, S1_S2F_Sep, S1I_S2F_Sep, S1IFS2FAll, S1IF_S2F_Sep, S1FS2FAll, S1F_S2F_Sep
NETWORK = 'calibration' #Options: Unet3, UnetFree, FusionS1S2Unet, QuadUnet, FullUnet, FullUnetWave, SwinTransStack, SwinUnet, FPB_Net, labelUnet, unet3Depth0, calibration

'''
DATA USAGE OPTIONS DESCRIPTION:
S1_I_only - Only Intensities values of the SAR image (Ivv and Ivh)
S1IV_only - Only Intensities values of the SAR image (Ivv and Ivh) + ratio of the intensities*****
S1_only - Coherence matrix of the SAR image (Ivv, Ivh, Re{C12},Im{C12})
S2_only - Only the 10m bands of the Sentinel-2 data
S1S2All - Coherence matrix of the SAR image concatenated with the 10m bands of the Sentinel-2 data
S1_S2_Sep - Coherence matrix of the SAR image and the 10m bands of the Sentinel-2 data in separate streams
S2S1IAll - Intensities values of the SAR image concatenated to the 10m bands of the Sentinel-2 data
S1I_S2_Sep - Intensities values of the SAR image and the 10m bands of the Sentinel-2 data in separate streams
S1IV_S2_Sep - Intensities values of the SAR image including the ration between them, and the 10m bands of the Sentinel-2 data in separate streams
S2F_only - 10+20m bands from the Sentinel-2 data
S1F_only - Coherence matrix of the SAR image (Ivv, Ivh, Re{C12},Im{C12}) + H/alpha decomposition features of the SAR image
S1IF_only - Intensities values of the SAR image + H/alpha decomposition features of the SAR image
S1IVF_only - Intensities values of the SAR image + ratio between intensities + H/alpha decomposition features of the SAR image*****
S1IS2FAll - Intensities values of the SAR image, concatenated to the 10+20m bands of Sentinel2
S1IFS2FAll - Intensities values of the SAR image + H/alpha decomposition features of the SAR image, concatenated to the 10+20m bands of Sentinel2
S1S2FAll - Coherence matrix of the SAR image, concatenated to the 10+20m bands of Sentinel2
S1FS2FAll - Coherence matrix of the SAR image (Ivv, Ivh, Re{C12},Im{C12}) + H/alpha decomposition features of the SAR image, concatenated to the 10+20m bands of Sentinel2
S1_S2F_Sep - Coherence matrix of the SAR image (Ivv, Ivh, Re{C12},Im{C12}), and the 10+20m bands of Sentinel2 in two separate streams
S1I_S2F_Sep - Intensities values of the SAR image, and the 10+20m bands of Sentinel2 in two separate streams
S1IF_S2F_Sep -  Intensities values of the SAR image + H/alpha decomposition features of the SAR image, and the 10+20m bands of Sentinel2 in two separate streams
S1F_S2F_Sep - Coherence matrix of the SAR image (Ivv, Ivh, Re{C12},Im{C12}) + H/alpha decomposition features of the SAR image, and the 10+20m bands of Sentinel2 in two separate streams
S1IVF_S2F_Sep - Intensities values of the SAR image+ ratio between intensities + H/alpha decomposition features of the SAR image, and the 10+20m bands of Sentinel2 in two separate streams
S1IVF_S2_Sep - Intensities values of the SAR image+ ratio between intensities + H/alpha decomposition features of the SAR image, and the 10m bands of Sentinel2 in two separate streams
S1I_S2Some -  Intensities values of the SAR image + 10m + Bands 6, 11 and 12 of Sentinel-2.
S1IV_S2Some - Intensities values of the SAR image+ ratio between intensities + 10m + Bands 6, 11 and 12 of Sentinel-2.
S1_Complex - intensities + using the complex part of the out of the diagonal as complex numbers.
S2_S1_Complex - S1 intensities + using the complex part of the out of the diagonal as complex numbers + all 10m bands of S2
'''


'''
NETWORK OPTIONS:

FullUnet - The Unet from the original paper (without the cutting parts and using paddings). 
Unet3 - A similar Unet but deepening only in 3 steps instead of 5.
UnetFree - A Unet-like architecture that you can build it with different backbones and different architectures, deepening as many steps as wanted, with as many convolution steps as wanted and all.
FusionS1S2Net - A double non-symmetric network for S1 and S2 fusion.
QuadUnet - use of the SLN network with th equadratic model.
FullUnetWave - Use of wavelet pooling
SwinTransStack - Use of Transformers  (tested, but needs improvement)
SwinUnet - Use of Transformers (tested, but needs improvement)
FPB_Net - Use of the full FPN_Network
labelUnet - proposed networks with two outputs. Suggest going to Baseline.py and comment/uncomment models with the two output you are interested in using.
unet3Depth0 - similar to Unet3
calibration - Calibration of one of the proposed network.
'''