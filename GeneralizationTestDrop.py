
import os
import numpy as np
import unetParams as params
from glob import glob
import tifffile
from osgeo import gdal
from MetricsCalc import *


'''
Function created to help make the original image padded by a buffer size.
mat - numpy array
size - size of the patch
'''
def Expand_size(mat, size):
	a = mat.shape
	x_max = a[0]+2*size
	y_max = a[1]+2*size
	if len(a)<3:
		envelopImg = np.zeros((x_max,y_max))
		envelopImg[size:a[0]+size,size:a[1]+size] = mat
		return envelopImg
	envelopImg = np.zeros((x_max,y_max,a[2]))
	envelopImg[size:a[0]+size,size:a[1]+size,:] = mat
	return envelopImg, x_max, y_max


'''
function to cut the original test image into patches to be fed into the network for inference.
image_path - address where the input test images are.
save_folder - folder to save the patches
type_data - if it is S1 or S2
'''
def PatchesCreate(image_path, save_folder, type_data):
	if (not os.path.isdir(save_folder)):
		try:
			os.makedirs(save_folder)
		except OSError:
			print ("Failed to create folder %s " % save_folder)
		else:
			print ("Created Folder %s" % save_folder)
	pad = int(params.IMG_SZ[0]/6)
	image = np.array(tifffile.imread(image_path))
	image, x_max, y_max = Expand_size(image , pad) #padding escolhido
	for i in range(0, x_max, params.IMG_SZ[0]-2*pad):
		for j in range(0, y_max, params.IMG_SZ[0]-2*pad):
			if i+params.IMG_SZ[0]<x_max:
				if j+params.IMG_SZ[1]<y_max:
					crop_input = image[i:i+params.IMG_SZ[0],j:j+params.IMG_SZ[1],:]
					tifffile.imwrite(save_folder+type_data+'_'+str(i)+"_"+ str(j)+ ".tif", crop_input, photometric='minisblack')
				else:
					crop_input = image[i:i+params.IMG_SZ[0],y_max-params.IMG_SZ[1]:y_max,:]
					tifffile.imwrite(save_folder+type_data+'_'+str(i)+"_"+ str(y_max-params.IMG_SZ[1])+ ".tif", crop_input, photometric='minisblack')
			else:
				if j+params.IMG_SZ[1]<y_max:
					crop_input = image[x_max-params.IMG_SZ[0]:x_max,j:j+params.IMG_SZ[1],:]
					tifffile.imwrite(save_folder+type_data+'_'+str(x_max-params.IMG_SZ[0])+"_"+ str(j)+ ".tif", crop_input, photometric='minisblack')
				else:
					crop_input = image[x_max-params.IMG_SZ[0]:x_max,y_max-params.IMG_SZ[1]:y_max,:]
					tifffile.imwrite(save_folder+type_data+'_'+str(x_max-params.IMG_SZ[0])+"_"+ str(y_max-params.IMG_SZ[1])+ ".tif", crop_input, photometric='minisblack')


'''
Function to reassample the inferred patches in to a final infered image compatible with the original test image. It saves the inference and all the other uncertainty metrics (MCLU, Entropy, Standard deviation, and 95% confidence interval)
original_image_path - path where the original image is
patches_Folder - pathe where the inferred patches are saved
save_folder - folder where to save the complete inferred image.
'''
def AssembleImage(original_image_path, patches_Folder, save_folder):
	if (not os.path.isdir(save_folder)):
		try:
			os.makedirs(save_folder)
		except OSError:
			print ("Failed to create folder %s " % save_folder)
		else:
			print ("Created Folder %s" % save_folder)

	pad = int(params.IMG_SZ[0]/6)
	patchesList = glob(os.path.join(patches_Folder, '%s*.%s' % (params.CLSPRED_FILE_STR,params.LABEL_FILE_EXT)))
	dataset = gdal.Open(original_image_path)
	width = dataset.RasterXSize
	height = dataset.RasterYSize
	x_max = width+2*pad
	y_max = height+2*pad
	pred_data = np.zeros((y_max, x_max))
	mclu_data = np.zeros((y_max, x_max))
	entropy_data = np.zeros((y_max, x_max))
	std_data = np.zeros((y_max, x_max))
	deltaP_data = np.zeros((y_max, x_max))
	for image in patchesList:
		imag =  np.array(tifffile.imread(image))
		mclu_ = np.array(tifffile.imread(image.replace(params.CLSPRED_FILE_STR,params.LABEL_MCLU_STR)))
		entr = np.array(tifffile.imread(image.replace(params.CLSPRED_FILE_STR,params.LABEL_ENT_STR)))
		std = np.array(tifffile.imread(image.replace(params.CLSPRED_FILE_STR,params.LABEL_STD_STR)))
		deltaP = np.array(tifffile.imread(image.replace(params.CLSPRED_FILE_STR,params.LABEL_DELPE_STR)))
		imageName = os.path.split(image)[-1]
		size = imag.shape
		parts = imageName.split('_')
		i=int(parts[1])
		j=int(parts[2].split('.')[0])
		pred_data[i+pad:i+size[0]-pad,j+pad:j+size[1]-pad]=imag[pad:params.IMG_SZ[0]-pad,pad:params.IMG_SZ[0]-pad]
		mclu_data[i+pad:i+size[0]-pad,j+pad:j+size[1]-pad]=mclu_[pad:params.IMG_SZ[0]-pad,pad:params.IMG_SZ[0]-pad]
		entropy_data[i+pad:i+size[0]-pad,j+pad:j+size[1]-pad]=entr[pad:params.IMG_SZ[0]-pad,pad:params.IMG_SZ[0]-pad]
		std_data[i+pad:i+size[0]-pad,j+pad:j+size[1]-pad]=std[pad:params.IMG_SZ[0]-pad,pad:params.IMG_SZ[0]-pad]
		deltaP_data[i+pad:i+size[0]-pad,j+pad:j+size[1]-pad]=deltaP[pad:params.IMG_SZ[0]-pad,pad:params.IMG_SZ[0]-pad]

	pred_data = pred_data[pad:height+pad,pad:width+pad]
	mclu_data = mclu_data[pad:height+pad,pad:width+pad]
	entropy_data = entropy_data[pad:height+pad,pad:width+pad]
	std_data=std_data[pad:height+pad,pad:width+pad]
	deltaP_data=deltaP_data[pad:height+pad,pad:width+pad]
	filename = os.path.split(original_image_path)[-1]
	filename = filename.replace(params.S1_FILE_STR, params.CLSPRED_FILE_STR)
	driver = gdal.GetDriverByName("GTiff")
	outdata = driver.Create(os.path.join(save_folder,filename), width, height, 1)
	outdata.SetGeoTransform(dataset.GetGeoTransform())  ##sets same geotransform as input
	outdata.SetProjection(dataset.GetProjection())  ##sets same projection as input
	outdata.GetRasterBand(1).WriteArray(pred_data)
	outdata.GetRasterBand(1).SetNoDataValue(10000)  ##if you want these values transparent
	outdata.FlushCache()  ##saves to disk!!
	outdata = None
	outdata1 = driver.Create(os.path.join(save_folder,filename.replace(params.CLSPRED_FILE_STR,params.LABEL_MCLU_STR)), width, height, 1, eType=gdal.GDT_Float32)
	outdata1.SetGeoTransform(dataset.GetGeoTransform())  ##sets same geotransform as input
	outdata1.SetProjection(dataset.GetProjection())  ##sets same projection as input
	outdata1.GetRasterBand(1).WriteArray(mclu_data)
	outdata1.GetRasterBand(1).SetNoDataValue(10000)  ##if you want these values transparent
	outdata1.FlushCache()  ##saves to disk!!
	outdata1 = None
	outdata2 = driver.Create(os.path.join(save_folder,filename.replace(params.CLSPRED_FILE_STR,params.LABEL_ENT_STR)), width, height, 1, eType=gdal.GDT_Float32)
	outdata2.SetGeoTransform(dataset.GetGeoTransform())  ##sets same geotransform as input
	outdata2.SetProjection(dataset.GetProjection())  ##sets same projection as input
	outdata2.GetRasterBand(1).WriteArray(entropy_data)
	outdata2.GetRasterBand(1).SetNoDataValue(10000)  ##if you want these values transparent
	outdata2.FlushCache()  ##saves to disk!!
	outdata2 = None
	outdata3 = driver.Create(os.path.join(save_folder,filename.replace(params.CLSPRED_FILE_STR,params.LABEL_STD_STR)), width, height, 1, eType=gdal.GDT_Float32)
	outdata3.SetGeoTransform(dataset.GetGeoTransform())  ##sets same geotransform as input
	outdata3.SetProjection(dataset.GetProjection())  ##sets same projection as input
	outdata3.GetRasterBand(1).WriteArray(std_data)
	outdata3.GetRasterBand(1).SetNoDataValue(10000)  ##if you want these values transparent
	outdata3.FlushCache()  ##saves to disk!!
	outdata3 = None
	outdata4 = driver.Create(os.path.join(save_folder,filename.replace(params.CLSPRED_FILE_STR,params.LABEL_DELPE_STR)), width, height, 1, eType=gdal.GDT_Float32)
	outdata4.SetGeoTransform(dataset.GetGeoTransform())  ##sets same geotransform as input
	outdata4.SetProjection(dataset.GetProjection())  ##sets same projection as input
	outdata4.GetRasterBand(1).WriteArray(deltaP_data)
	outdata4.GetRasterBand(1).SetNoDataValue(10000)  ##if you want these values transparent
	outdata4.FlushCache()  ##saves to disk!!
	outdata3 = None
	band = None
	ds = None


'''
Function to delete intermediate files 
path_to_files - Folder where the .tif intermediate files are.
'''
def deleteInterFiles(path_to_files):
	files = glob(os.path.join(path_to_files,'*.tif'))
	for f in files:
		os.remove(f)

'''
Function that perform the generalization test on the test images.
It performs a statistical test based on the MC Dropout and saves the images based on different approaches to combine the MC Dropout realizations (mean, median, voting or percentile)
'''

def GenTest():
	
	#Get all imags in testing folders
	S1_img_list = glob(os.path.join(params.ORG_TEST_DIR+'S1/', '*%s*.%s' % (params.S1_FILE_STR,params.S1_FILE_EXT)))

	for i in S1_img_list:
		#create the 256x256 files for the testing (depending on the type of data)
		filename = os.path.split(i)[-1]
		filename = filename.replace(params.S1_FILE_STR, params.S2_FILE_STR)
		filename = filename.replace(params.S1_FILE_EXT, params.S2_FILE_EXT)
		S2_i = os.path.join(params.ORG_TEST_DIR+'S2/',filename)
		PatchesCreate(i, params.TEST_DIR, 'S1')
		PatchesCreate(S2_i, params.TEST_DIR, 'S2')
		#test in the network
		os.system('python NetworkRun.py droptest')
		#regroup into the original image size with the image's georreference and save data
		AssembleImage(i, params.OUTPUT_DIR+'mean/', params.ASSEMBLE_OUT_DIR+'mean/')
		AssembleImage(i, params.OUTPUT_DIR+'median/', params.ASSEMBLE_OUT_DIR+'median/')
		AssembleImage(i, params.OUTPUT_DIR+'voting/', params.ASSEMBLE_OUT_DIR+'voting/')
		AssembleImage(i, params.OUTPUT_DIR+'percentile/', params.ASSEMBLE_OUT_DIR+'percentile/')
		#delete 256x256 patches created and 256x256 patches infered by the network
		deleteInterFiles(params.TEST_DIR)
		deleteInterFiles(params.OUTPUT_DIR+'percentile/')
		deleteInterFiles(params.OUTPUT_DIR+'mean/')
		deleteInterFiles(params.OUTPUT_DIR+'median/')
		deleteInterFiles(params.OUTPUT_DIR+'voting/')
	

	metrics = MetricsCalc('./GenTest/GT_L1/', params.ASSEMBLE_OUT_DIR+'mean/', './GenTest/S1/','./GenTest/S2/')
	metrics.StatsOnFile('stats.txt')
	print(metrics.confMatrix)
	metrics = MetricsGroupImg('./GenTest/GT_L1/', params.ASSEMBLE_OUT_DIR+'mean/', './GenTest/S1/','./GenTest/S2/')
	metrics.StatsOnFile('stats_perFile.txt')

	metrics = MetricsCalc('./GenTest/GT_L1/', params.ASSEMBLE_OUT_DIR+'median/', './GenTest/S1/','./GenTest/S2/')
	metrics.StatsOnFile('stats.txt')
	print(metrics.confMatrix)
	metrics = MetricsGroupImg('./GenTest/GT_L1/', params.ASSEMBLE_OUT_DIR+'median/', './GenTest/S1/','./GenTest/S2/')
	metrics.StatsOnFile('stats_perFile.txt')

	metrics = MetricsCalc('./GenTest/GT_L1/', params.ASSEMBLE_OUT_DIR+'voting/', './GenTest/S1/','./GenTest/S2/')
	metrics.StatsOnFile('stats.txt')
	print(metrics.confMatrix)
	metrics = MetricsGroupImg('./GenTest/GT_L1/', params.ASSEMBLE_OUT_DIR+'voting/', './GenTest/S1/','./GenTest/S2/')
	metrics.StatsOnFile('stats_perFile.txt')

	metrics = MetricsCalc('./GenTest/GT_L1/', params.ASSEMBLE_OUT_DIR+'percentile/', './GenTest/S1/','./GenTest/S2/')
	metrics.StatsOnFile('stats.txt')
	print(metrics.confMatrix)
	metrics = MetricsGroupImg('./GenTest/GT_L1/', params.ASSEMBLE_OUT_DIR+'percentile/', './GenTest/S1/','./GenTest/S2/')
	metrics.StatsOnFile('stats_perFile.txt')