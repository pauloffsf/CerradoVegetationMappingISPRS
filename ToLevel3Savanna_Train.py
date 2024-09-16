from datafunctions import *
from glob import glob
import unetParams as params
from osgeo import gdal


'''
Function to georreference the numpy array
data: numpy array
filename: name of the tiff file to be saved
metadata: address to the original tiff file that the numpy array is also referenced to geographically.
'''
def georrefData(data, filename, metadata):
	dataset = gdal.Open(metadata)
	width = dataset.RasterXSize
	height = dataset.RasterYSize
	driver = gdal.GetDriverByName("GTiff")
	outdata = driver.Create(filename, width, height, 1)
	outdata.SetGeoTransform(dataset.GetGeoTransform())  ##sets same geotransform as input
	outdata.SetProjection(dataset.GetProjection())  ##sets same projection as input
	outdata.GetRasterBand(1).WriteArray(data)
	outdata.FlushCache()  ##saves to disk!!
	outdata = None
	band = None
	ds = None


'''
Function that changes the classes of the Level-3 vegetation map to the
training classes for Savannah Types. Function is called within ToLevel3_TrainSavanna

img: numpy array with the original class numbers
@return: another numpy array with the corrected class numbers
'''
def changeClass(img):

	newImg = np.zeros_like(img)
	#Antropic
	A = img == 1
	newImg = 0*A+newImg
	#Water
	A = img == 2
	newImg = 0*A+newImg
	#Natural not vegetated
	A = img == 3
	newImg = newImg + 0*A
	#Grassland
	A = img == 4
	newImg = newImg + 0*A
	A = img == 5
	newImg = newImg + 0*A
	A = img == 6
	newImg = newImg + 0*A
	#Forest
	A = img == 7
	newImg = newImg + 0*A
	A = img == 8
	newImg = newImg + 0*A
	A = img == 9
	newImg = newImg + 0*A
	A = img == 10
	newImg = newImg + 0*A
	#Savannah
	A = img == 11
	newImg = newImg + 1*A
	A = img == 12
	newImg = newImg + 2*A
	A = img == 13
	newImg = newImg + 3*A
	A = img == 14
	newImg = newImg + 4*A
	#Vegetacao secundaria
	A = img == 15
	newImg = newImg + 0*A


	return newImg

'''
Function that gathers all the tiff files in a folder to transform the classes in level-3 to the correct class numbers 
for the training the Savannah types. This fuction calls changeClass within.

end: address where all the reference level-3 labeled images are
saveRef: boolean to decide if the georreference is gonna transmitted to the new images or not.
'''
def ToLevel3_TrainSavanna(end, saveRef=False):
	list_img = glob(os.path.join(end, '*.%s' % (params.LABEL_FILE_EXT)))
	for i in list_img:
		img = load_img(i)
		img = changeClass(img)
		if saveRef:
			georrefData(img, i, i)
		else:
			tifffile.imwrite(i, img)
