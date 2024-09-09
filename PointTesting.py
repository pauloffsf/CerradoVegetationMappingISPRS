import numpy as np
from osgeo import gdal
from osgeo import ogr
from osgeo import osr
from glob import glob
import os
import tifffile
import unetParams as params


'''
Script to test the point-wise compariison with the Bendini Fieldwork Points. There are three types of comparison.
All three types are called from within the function test_Images_Points, therefore, you only need to call it.
'''

'''
function to save a shapefile of alist of GeoPoints
total_points - list of points
str_filename - name of the shapefile to be saved.
'''
def saveshapefile(total_points, str_filename):
    drivershape = ogr.GetDriverByName('ESRI Shapefile')
    dataset = drivershape.CreateDataSource(str_filename+'.shp')
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    CamadaNova=dataset.CreateLayer(str_filename,srs,ogr.wkbPoint)
    field_class = ogr.FieldDefn('Classe', ogr.OFTString)
    field_class.SetWidth(24)
    CamadaNova.CreateField(field_class)
    for feature in total_points:
        Novoitem = ogr.Feature(CamadaNova.GetLayerDefn())
        classe = feature.GetField(6)
        Novoitem.SetField('Classe', classe)
        wkt = feature.GetField(0)
        geometry = ogr.CreateGeometryFromWkt(wkt)
        Novoitem.SetGeometry(geometry)
        CamadaNova.CreateFeature(Novoitem)
        Novoitem = None
    dataset = None


'''
Function to load an image
imgPath - path to the image file
'''
def load_img(imgPath):
	'''
	Load image
	:param imgPath: path of the image to load
	:return: numpy array of the image
	'''
	if imgPath.endswith('.tif'):
		img = tifffile.imread(imgPath)
	else:
		img = np.array(cv2.imread(imgPath))
	return img


'''
Function to write a WKT string 
latmin - Min value of latitude
latmax - Max value of latitude
longmin - Min value of longitude
longmax - Max value of longitude
'''
def WKTString (latmin, latmax, longmin, longmax):

	return 'POLYGON(('+str(longmin)+' '+str(latmin)+', '+str(longmin)+' '+str(latmax)+', '+str(longmax)+' '+str(latmax)+', '+str(longmax)+' '+str(latmin)+', '+str(longmin)+' '+str(latmin)+'))'



'''
function that checks the amount of classes around a certain numpy array window.
around - the numpy window to be tested
total_classes - the total number of classes to be tested
'''
def aroundTest(around, total_classes):
    total_values  = np.zeros(total_classes)
    for i in range(total_classes):
        total_values[i]=np.sum(around==(i))
    return np.argmax(total_values), total_values

'''
Function that looks if a certain class is within a window around a certain point.
Value of the window must be set within the function (it is not a parameter. Here it is set for a 5x5 window.)
sentinelImg - filename of the image to be tested
csvPoints - filename of the file with the points to be tested.
'''
def PointTestingHasAround(sentinelImg, csvPoints):

    endS1 = './GenTest/S1'
    endS2 = './GenTest/S2'
    endGT = './GenTest/GT_L2'
    filename = os.path.split(sentinelImg)[-1]
    S1_img = load_img(os.path.join(endS1, filename.replace(params.CLSPRED_FILE_STR,'S1')))
    S2_img = load_img(os.path.join(endS2, filename.replace(params.CLSPRED_FILE_STR,'S2')))
    GT_img = load_img(os.path.join(endGT, filename.replace(params.CLSPRED_FILE_STR,'GT')))
    #open image:
    sentinel_img = gdal.Open(sentinelImg)
    band = sentinel_img.GetRasterBand(1)  # bands start at one
    img = band.ReadAsArray().astype(np.float16)

    A = S1_img != [0,0,0,0,0,0,0]
    A = A[:,:,0]
    B = S2_img != [0,0,0,0,0,0,0,0,0,0]
    B = B[:,:,0]
    C = GT_img!=0
    A = A*B*C
    img = np.multiply(img,A)

    geoTransform = sentinel_img.GetGeoTransform()
    x_0 = geoTransform[0]
    y_0 = geoTransform[3]
    delta_x = geoTransform[1]
    delta_y = geoTransform[5]
    
    #open csv
    file = ogr.Open(csvPoints)
    shapefile = file.GetLayer(0)
    total_features = shapefile.GetFeatureCount()
    shape_confMatrix = (7,7)
    confusion_matrix = np.zeros(shape_confMatrix)
    points_list = []
    errors_points_list = []
    for feature in shapefile:
        point = feature.GetField(0)
        x = point.split('(')[-1]
        x = float(x.split(' ')[0])
        y = point.split(')')[0]
        y = float(y.split(' ')[-1])
        x = int((x - x_0)/delta_x)
        y = int((y - y_0)/delta_y)
        inference, vector = aroundTest(img[y-2:y+3,x-2:x+3], 8) #Here you can change the size of the window
        classe = int(feature.GetField(6))
        if classe == 1:
            classe = 6
        elif classe == 2:
            classe = 4
        elif classe == 3:
            classe = 5
        if (vector[classe]>0):
            inference = classe
        if x>0 and x<img.shape[1]:
            if y>0 and y<img.shape[0]:
                if img[y,x]>0:
                    confusion_matrix[classe-1,int(inference-1)]=confusion_matrix[classe-1,int(inference-1)]+1
                    points_list.append(feature)
                    if classe!=inference:
                        errors_points_list.append(feature)
    return errors_points_list, points_list, confusion_matrix

'''
Function that looks if a certain class is the most voted class within a window around a certain point.
Value of the window must be set within the function (it is not a parameter. Here it is set for a 5x5 window.)
sentinelImg - filename of the image to be tested
csvPoints - filename of the file with the points to be tested.
'''
def PointTestingAround(sentinelImg, csvPoints):

    endS1 = './GenTest/S1'
    endS2 = './GenTest/S2'
    endGT = './GenTest/GT_L2'
    filename = os.path.split(sentinelImg)[-1]
    S1_img = load_img(os.path.join(endS1, filename.replace(params.CLSPRED_FILE_STR,'S1')))
    S2_img = load_img(os.path.join(endS2, filename.replace(params.CLSPRED_FILE_STR,'S2')))
    GT_img = load_img(os.path.join(endGT, filename.replace(params.CLSPRED_FILE_STR,'GT')))
    #open image:
    sentinel_img = gdal.Open(sentinelImg)
    band = sentinel_img.GetRasterBand(1)  # bands start at one
    img = band.ReadAsArray().astype(np.float16)

    A = S1_img != [0,0,0,0,0,0,0]
    A = A[:,:,0]
    B = S2_img != [0,0,0,0,0,0,0,0,0,0]
    B = B[:,:,0]
    C = GT_img!=0
    A = A*B*C
    img = np.multiply(img,A)

    geoTransform = sentinel_img.GetGeoTransform()
    x_0 = geoTransform[0]
    y_0 = geoTransform[3]
    delta_x = geoTransform[1]
    delta_y = geoTransform[5]
    
    #open csv
    file = ogr.Open(csvPoints)
    shapefile = file.GetLayer(0)
    total_features = shapefile.GetFeatureCount()
    shape_confMatrix = (7,7)
    confusion_matrix = np.zeros(shape_confMatrix)
    points_list = []
    errors_points_list = []
    for feature in shapefile:
        point = feature.GetField(0)
        x = point.split('(')[-1]
        x = float(x.split(' ')[0])
        y = point.split(')')[0]
        y = float(y.split(' ')[-1])
        x = int((x - x_0)/delta_x)
        y = int((y - y_0)/delta_y)
        inference, vector = aroundTest(img[y-2:y+3,x-2:x+3], 8) #Here you can change the size of the window
        classe = int(feature.GetField(6))
        if classe == 1:
            classe = 6
        elif classe == 2:
            classe = 4
        elif classe == 3:
            classe = 5
        if x>0 and x<img.shape[1]:
            if y>0 and y<img.shape[0]:
                if img[y,x]>0:
                    confusion_matrix[classe-1,int(inference-1)]=confusion_matrix[classe-1,int(inference-1)]+1
                    points_list.append(feature)
                    if classe!=inference:
                        errors_points_list.append(feature)
    return errors_points_list, points_list, confusion_matrix


'''
Function that looks if a certain class is in a geolocated point-based reference.
sentinelImg - filename of the image to be tested
csvPoints - filename of the file with the points to be tested.
'''
def PointTesting(sentinelImg, csvPoints):

    endS1 = './GenTest/S1'
    endS2 = './GenTest/S2'
    endGT = './GenTest/GT_L2'
    filename = os.path.split(sentinelImg)[-1]
    S1_img = load_img(os.path.join(endS1, filename.replace(params.CLSPRED_FILE_STR,'S1')))
    S2_img = load_img(os.path.join(endS2, filename.replace(params.CLSPRED_FILE_STR,'S2')))
    GT_img = load_img(os.path.join(endGT, filename.replace(params.CLSPRED_FILE_STR,'GT')))
    #open image:
    sentinel_img = gdal.Open(sentinelImg)
    band = sentinel_img.GetRasterBand(1)  # bands start at one
    img = band.ReadAsArray().astype(np.float16)

    A = S1_img != [0,0,0,0,0,0,0]
    A = A[:,:,0]
    B = S2_img != [0,0,0,0,0,0,0,0,0,0]
    B = B[:,:,0]
    C = GT_img!=0
    A = A*B*C
    img = np.multiply(img,A)
    geoTransform = sentinel_img.GetGeoTransform()
    x_0 = geoTransform[0]
    y_0 = geoTransform[3]
    delta_x = geoTransform[1]
    delta_y = geoTransform[5]
    
    #open csv
    file = ogr.Open(csvPoints)
    shapefile = file.GetLayer(0)
    total_features = shapefile.GetFeatureCount()
    shape_confMatrix = (7,7)
    confusion_matrix = np.zeros(shape_confMatrix)
    points_list = []
    errors_points_list = []
    for feature in shapefile:
        point = feature.GetField(0)
        x = point.split('(')[-1]
        x = float(x.split(' ')[0])
        y = point.split(')')[0]
        y = float(y.split(' ')[-1])
        x = int((x - x_0)/delta_x)
        y = int((y - y_0)/delta_y)
        classe = int(feature.GetField(6))
        if classe == 1:
            classe = 6
        elif classe == 2:
            classe = 4
        elif classe == 3:
            classe = 5
        if x>0 and x<img.shape[1]:
            if y>0 and y<img.shape[0]:
                if img[y,x]>0:
                    confusion_matrix[classe-1,int(img[y,x]-1)]=confusion_matrix[classe-1,int(img[y,x]-1)]+1
                    points_list.append(feature)
                    if classe!=img[y,x]:
                        errors_points_list.append(feature)
    return errors_points_list, points_list, confusion_matrix


'''
main function to be called from this script. it will test all three types of point comparison. and save all the results in different shapefiles.
end_test - address where all the test images (level-2) are.
'''
def test_Images_Points(end_test):

    shape_confMatrix = (7,7)
    confusion_matrix_single = np.zeros(shape_confMatrix)
    confusion_matrix_around = np.zeros(shape_confMatrix)
    confusion_matrix_hasaround = np.zeros(shape_confMatrix)
    total_points=[]
    errors_point_exact = []
    errors_point_around = []
    errors_point_hasaround = []
    list_img = glob(os.path.join(end_test, '*%s*.%s' % (params.CLSPRED_FILE_STR,params.LABEL_FILE_EXT)))
    print(len(list_img))
    csv_file = './Samples_for_Vegetation_Mapping_Bendini-etal_2021.csv'
    for i in list_img:
        print(i)
        errors_pointsi, list_pointsi, conf_mati = PointTesting(i,csv_file)
        confusion_matrix_single = confusion_matrix_single+conf_mati
        total_points = total_points+list_pointsi
        errors_point_exact = errors_point_exact+errors_pointsi
        errors_pointsi,list_pointsi, conf_mati = PointTestingAround(i,csv_file)
        confusion_matrix_around = confusion_matrix_around+conf_mati
        errors_point_around=errors_point_around+errors_pointsi
        errors_pointsi, list_pointsi, conf_mati = PointTestingHasAround(i,csv_file)
        confusion_matrix_hasaround = confusion_matrix_hasaround+conf_mati
        errors_point_hasaround = errors_point_hasaround+errors_pointsi

    print(confusion_matrix_single)
    print(confusion_matrix_around)
    print(confusion_matrix_hasaround)
    print(np.sum(confusion_matrix_around))
    saveshapefile(total_points, 'PontosBendiniTest')
    saveshapefile(errors_point_exact, 'ErrorPontosBendiniExact')
    saveshapefile(errors_point_around, 'ErrorPontosBendiniAround')
    saveshapefile(errors_point_hasaround, 'ErrorPontosBendiniHasAround')