from datafunctions import *
from glob import glob
import unetParams as params
from osgeo import gdal
from MetricsCalc import *

'''
Script to Match the first and second level inferences

We use it by calling the functions fullMatch and matchUncertainty only.

'''


'''
georrefData:
Function to create the final image with the correct corresponding georreference.
data: numpy array to be saved as a TIFF image
filename: name of the TIFF file to be saved.
metadata: reference TIFF file which has the georreference that will be used.
'''

def georrefData(data, filename, metadata, data_type = None):
	dataset = gdal.Open(metadata)
	width = dataset.RasterXSize
	height = dataset.RasterYSize
	driver = gdal.GetDriverByName("GTiff")
	if data_type is None:
		outdata = driver.Create(filename, width, height, 1)
	else:
		outdata = driver.Create(filename, width, height, 1, eType=gdal.GDT_Float32)
	outdata.SetGeoTransform(dataset.GetGeoTransform())  ##sets same geotransform as input
	outdata.SetProjection(dataset.GetProjection())  ##sets same projection as input
	outdata.GetRasterBand(1).WriteArray(data)
	outdata.FlushCache()  ##saves to disk!!
	outdata = None
	band = None
	ds = None



'''
function that matches the lvl-1 and lvl-2 inference images and return the complete inference. This function is called in the fullMatch function
clsLvl1 - numpy array image with the lvl-1 inference
clsLvl2 - numpy array image with the lvl-2 inference
'''

def match(clsLvl1, clsLvl2):

	newImg = np.zeros_like(clsLvl1)

	#Antropic
	A = clsLvl1 == 1
	newImg = newImg + 1*A
	#water
	A = clsLvl1 == 2
	newImg = newImg + 2*A
	
	#natural non vegetated
	A = clsLvl1 == 3
	B = clsLvl2 ==1
	C=A*B
	newImg = newImg + 3*C

	#grassland
	B = clsLvl2 ==2
	C=A*B
	newImg = newImg + 4*C

	#forest
	B = clsLvl2 ==3
	C=A*B
	newImg = newImg + 5*C

	#savannah
	B = clsLvl2 ==4
	C=A*B
	newImg = newImg + 6*C

	#secondary vegetation
	B = clsLvl2 ==5
	C=A*B
	newImg = newImg + 7*C

	return newImg


'''
Function that searches for all the lvl-1 and lvl-2 inference images and combine the respective ones. This function will call match.
end_CLS1 - address where the images of the first level inference are
end_CLS2 - address where the images of the second level inference are
endsave - address where you want to save the combined inference images 
'''

def fullMatch(end_CLS1, end_CLS2, endsave):

	listCLS1 = glob(os.path.join(end_CLS1, '*%s*.%s' % (params.CLSPRED_FILE_STR, params.LABEL_FILE_EXT)))

	if not os.path.isdir(endsave):
		os.makedirs(endsave)

	for imgCLS1 in listCLS1:
		filename = os.path.split(imgCLS1)[-1]
		clsLvl1 = load_img(imgCLS1)
		clsLvl2 = load_img(end_CLS2+filename)
		new_img = match(clsLvl1,clsLvl2)
		georrefData(new_img, endsave+filename, imgCLS1)


'''
Function used to combine the two uncertainty numpy array images into one numpy array (this is called from matchUncertainty)
img - numpy array image with the secon-level inference results
uncertlvl1 - numpy array image with the uncertainty at Level-1
uncertlvl2 - numpy array image with the uncertainty at Level-2
'''
def matchUncert(img, uncertlvl1, uncertlvl2):
	newImg = np.zeros_like(img)
	newImg = newImg.astype(float)

	a = img <= 2
	newImg = newImg+a.astype(float)*uncertlvl1
	b = img > 2
	newImg = newImg+b.astype(float)*uncertlvl2

	return newImg


'''
Function that searches for all the lvl-1 and lvl-2 uncertainty images and combine the respective ones. This function will call matchUncert.
savedComb - address where you want to save the combined uncertainty images 
end_Unlvl1 - address where the images of the first level uncertainties are
end_Unlvl2 - address where the images of the second level uncertainties are
'''
def matchUncertainty(savedComb, end_Unlvl1, end_Unlvl2):

	listComb = glob(os.path.join(savedComb, '*%s*.%s' % (params.CLSPRED_FILE_STR, params.LABEL_FILE_EXT)))


	for imgadd in listComb:
		filename = os.path.split(imgadd)[-1]
		img = load_img(imgadd)
		uncertlvl1 = load_img(os.path.join(end_Unlvl1,filename.replace(params.CLSPRED_FILE_STR,params.LABEL_STD_STR)))
		uncertlvl1 = uncertlvl1/maxUnlvl1
		uncertlvl2 = load_img(os.path.join(end_Unlvl2,filename.replace(params.CLSPRED_FILE_STR,params.LABEL_STD_STR)))
		uncertlvl2 = uncertlvl2/maxUnlvl2

		newImg = matchUncert(img, uncertlvl1, uncertlvl2)
		georrefData(newImg, os.path.join(savedComb,filename.replace(params.CLSPRED_FILE_STR,params.LABEL_STD_STR)), imgadd, data_type=1)