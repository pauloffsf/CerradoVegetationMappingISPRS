
import numpy as np 
import os
from glob import glob
from sklearn.ensemble import RandomForestClassifier
from datafunctions import *
import random as rng
from osgeo import gdal
from osgeo import ogr
from osgeo import osr
import pandas as pd
import string
import pickle
from tqdm import tqdm

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


def ndviF(nir, red):
	a = nir-red
	b = nir+red
	return np.divide(a,b,out=np.zeros_like(a),where=b!=0)


def evi2F(nir, red):
	a = 2.5*(nir-red)
	b = nir+2.4*red+1
	return np.divide(a,b,out=np.zeros_like(a),where=b!=0)

def saviF(nir, red):
	a = 1.5*(nir-red)
	b = nir+red+0.5
	return np.divide(a,b,out=np.zeros_like(a),where=b!=0)

def swir21F(swir1, swir2):
	return np.divide(swir2,swir1,out=np.zeros_like(swir2),where=swir1!=0)


def vv_vh_ratioF(ivv, ivh):

	ivv= np.power(10,ivv/10)
	ivh = np.power(10,ivh/10)

	return np.divide(ivh,ivv,out=np.zeros_like(ivv),where=ivv!=0)

def ndwiF(green, nir):
	a = green-nir
	b = green+nir
	return np.divide(a,b,out=np.zeros_like(a),where=b!=0)

def findRandom(GT, classe):
	A = GT==classe
	class_possibility = np.argwhere(A)
	random_idx = rng.randint(0,len(class_possibility)-1)
	return class_possibility[random_idx]

def Textura(x,y,img,window):

	sub_img = img[max(0,x-window):min(x+window,img.shape[0]),max(0,y-window):min(y+window,img.shape[1])]
	return np.var(sub_img)


def createFeatureSpace(list_points, S2_wet, S2_dry, S2_ann, S2_annVar, S1_wet, S1_dry, S1_ann, S1_annVar, slope):

	features_space = []

	Indvi_wet = ndviF(S2_wet[:,:,3],S2_wet[:,:,2])
	Ievi2_wet = evi2F(S2_wet[:,:,3],S2_wet[:,:,2])
	ISavi_wet = saviF(S2_wet[:,:,3],S2_wet[:,:,2])
	Iswir_wet = swir21F(S2_wet[:,:,8],S2_wet[:,:,9])
	vh_vv_wet = vv_vh_ratioF(S1_wet[:,:,0],S1_wet[:,:,1])

	Indvi_dry = ndviF(S2_dry[:,:,3],S2_dry[:,:,2])
	Ievi_dry = evi2F(S2_dry[:,:,3],S2_dry[:,:,2])
	Isavi_dry = saviF(S2_dry[:,:,3],S2_dry[:,:,2])
	Iswir_dry = swir21F(S2_dry[:,:,8],S2_dry[:,:,9])
	vh_vv_dry = vv_vh_ratioF(S1_dry[:,:,0],S1_dry[:,:,1])

	Indvi_ann = ndviF(S2_ann[:,:,3],S2_ann[:,:,2])
	Ievi_ann = evi2F(S2_ann[:,:,3],S2_ann[:,:,2])
	Isavi_ann = saviF(S2_ann[:,:,3],S2_ann[:,:,2])
	Iswir_ann = swir21F(S2_ann[:,:,8],S2_ann[:,:,9])
	vh_vv_ann = vv_vh_ratioF(S1_ann[:,:,0],S1_ann[:,:,1])

	Indvi_annV = ndviF(S2_annVar[:,:,3],S2_annVar[:,:,2])
	Ievi_annV = evi2F(S2_annVar[:,:,3],S2_annVar[:,:,2])
	Isavi_annV = saviF(S2_annVar[:,:,3],S2_annVar[:,:,2])
	Iswir_annV = swir21F(S2_annVar[:,:,8],S2_annVar[:,:,9])


	Indwi = ndwiF(S2_ann[:,:,1],S2_ann[:,:,3])

	for point in list_points:
		x=point[0]
		y=point[1]
		Owet = S2_wet[x,y,:] #10
		Odry = S2_dry[x,y,:] #10
		OAnn = S2_ann[x,y,:] #10
		Sarwet = S1_wet[x,y,:] #2
		Sardry = S1_dry[x,y,:] #2
		Sarann = S1_ann[x,y,:] #2
		SarannVar = S1_annVar[x,y,:]#2
		OAnnVar = S2_annVar[x,y,2:8] #6
		ndvi_wet = Indvi_wet[x,y] #1
		evi2_wet = Ievi2_wet[x,y] #1
		savi_wet = ISavi_wet[x,y] #1
		swir21_wet = Iswir_wet[x,y] #1
		vv_vh_wet = vh_vv_wet[x,y] #1
		ndvi_dry = Indvi_dry[x,y] #1
		evi2_dry = Ievi_dry[x,y] #1
		savi_dry = Isavi_dry[x,y] #1
		swir21_dry = Iswir_dry[x,y] #1
		vv_vh_dry = vh_vv_dry[x,y] #1
		ndvi_ann = Indvi_ann[x,y] #1
		evi2_ann = Ievi_ann[x,y] #1
		savi_ann = Isavi_ann[x,y] #1
		swir21_ann = Iswir_ann[x,y] #1
		vv_vh_ann = vh_vv_ann[x,y] #1
		ndvi_annV = Indvi_annV[x,y] #1
		evi2_annV = Ievi_annV[x,y] #1
		savi_annV = Isavi_annV[x,y] #1
		swir21_annV = Iswir_annV[x,y] #1
		ndwi = Indwi[x,y] #1
		dem_sl = slope[x,y] #1
		evi_T_wet = Textura(x,y,Ievi2_wet,5) #1
		evi_T_dry = Textura(x,y,Ievi_dry,5) #1
		ndvi_T_wet = Textura(x,y,Indvi_wet,5) #1
		ndvi_T_dry = Textura(x,y,Indvi_dry,5) #1
		savi_T_wet = Textura(x,y,ISavi_wet,5) #1
		savi_T_dry = Textura(x,y,Isavi_dry,5) #1

		features_space.append(np.concatenate([Owet,Odry,OAnn,OAnnVar,Sarwet,Sardry,Sarann,SarannVar,np.array([ndvi_wet,evi2_wet,savi_wet,swir21_wet,vv_vh_wet,
							ndvi_dry, evi2_dry, savi_dry, swir21_dry, vv_vh_dry, ndvi_ann, evi2_ann, savi_ann, swir21_ann, vv_vh_ann,
							ndvi_annV, evi2_annV, savi_annV, swir21_annV, ndwi, dem_sl, evi_T_wet, evi_T_dry, ndvi_T_wet, ndvi_T_dry, 
							savi_T_wet,savi_T_dry])]))

	return np.array(features_space)

def TestCreateFeatureSpace():

	points = [(10,15), (12,16)]
	S2_wet=load_img('./RF_train/Q1/Q1_S2_Wet.tif')
	S2_dry=load_img('./RF_train/Q1/Q1_S2_Dry.tif')
	S2_ann=load_img('./RF_train/Q1/Q1_S2_Ann.tif')
	S2_annVar=load_img('./RF_train/Q1/Q1_S2_AnnV.tif')
	S1_wet=load_img('./RF_train/Q1/Q1_S1_Wet.tif')
	S1_dry=load_img('./RF_train/Q1/Q1_S1_Dry.tif')
	S1_ann=load_img('./RF_train/Q1/Q1_S1_Ann.tif')
	S1_annVar=load_img('./RF_train/Q1/Q1_S1_AnnV.tif')
	slope=load_img('./RF_train/Q1/Q1_Slope.tif')
	feature = createFeatureSpace(points, S2_wet, S2_dry, S2_ann, S2_annVar, S1_wet, S1_dry, S1_ann, S1_annVar, slope)
	print(feature)
	print(feature.shape)

def chooseRandomPoints(amount_points_perclass, burnAreaMask, reference, n_classes):
	shape = burnAreaMask.shape

	chosenpoints = []

	for i in range(1,n_classes):
		A = reference==i
		A=A*(burnAreaMask==0)
		possible_points_perclass = np.argwhere(A)
		if (len(possible_points_perclass)>amount_points_perclass):
			selected = rng.sample(range(len(possible_points_perclass)),amount_points_perclass)
			possible_points_perclass=possible_points_perclass[selected]
		chosenpoints.append(possible_points_perclass)

	chosenpoints = np.concatenate(chosenpoints)
	indexes = np.arange(chosenpoints.shape[0])
	np.random.shuffle(indexes)
	chosenpoints = chosenpoints[indexes]
	return chosenpoints


def testChooseRandom(amount_points_perclass, burnAreaMask_str, reference_str, n_classes):

	BA = load_img(burnAreaMask_str)
	GT = load_img(reference_str)

	points = chooseRandomPoints(1000, BA,GT,8)
	print(points)
	print(len(points))

def createTrainingDataset(root_add, csv_save_file):

	regions = ['Q1','Q2','Q3','Q4','Q5','Q6','Q7','Q8','Q9','Q10']
	file_type = ['GT.tif', 'Slope.tif', 'S1_Ann.tif', 'S1_AnnV.tif', 'S1_Dry.tif','S1_Wet.tif','S2_Ann.tif','S2_AnnV.tif','S2_Dry.tif','S2_Wet.tif','BA.tif']
	template_add = string.Template(root_add+'${region}/${region}_')

	features = []
	labels = []

	for i in regions:
		print(i)
		img_gt = load_img(template_add.substitute(region=i)+file_type[0])
		img_slope = load_img(template_add.substitute(region=i)+file_type[1])
		img_s1_ann = load_img(template_add.substitute(region=i)+file_type[2])
		img_s1_annV = load_img(template_add.substitute(region=i)+file_type[3])
		img_s1_dry = load_img(template_add.substitute(region=i)+file_type[4])
		img_s1_wet = load_img(template_add.substitute(region=i)+file_type[5])
		img_s2_ann = load_img(template_add.substitute(region=i)+file_type[6])
		img_s2_annV = load_img(template_add.substitute(region=i)+file_type[7])
		img_s2_dry = load_img(template_add.substitute(region=i)+file_type[8])
		img_s2_wet = load_img(template_add.substitute(region=i)+file_type[9])
		img_BA = load_img(template_add.substitute(region=i)+file_type[10])

		points = chooseRandomPoints(1000, img_BA, img_gt, 8)

		features.append(createFeatureSpace(points, img_s2_wet, img_s2_dry,
												img_s2_ann, img_s2_annV, img_s1_wet,
												img_s1_dry, img_s1_ann, img_s1_annV, img_slope))
		for point in points:
			labels.append(img_gt[point[0],point[1]])

	features = np.concatenate(features)
	labels = np.array(labels)
	print(features.shape)
	print(labels.shape)

	df_features = pd.DataFrame(features)
	df_labels = pd.DataFrame(labels)

	df_features.to_csv(os.path.join(root_add,'features_'+csv_save_file), index=False)
	df_labels.to_csv(os.path.join(root_add,'labels_'+csv_save_file), index=False)


#createTrainingDataset('./RF_train/', 'RF_Cerrado.csv')

def RandomForest_Train(csv_features, csv_labels, save_model_name):
	df_features = pd.read_csv(csv_features)
	df_labels = pd.read_csv(csv_labels)

	x_train = df_features.to_numpy()
	print(x_train.shape)
	y_train = df_labels.to_numpy()
	print(y_train.shape)
	rf = RandomForestClassifier()
	rf.fit(x_train, y_train.ravel())

	with open(save_model_name, 'wb') as f:
		pickle.dump(rf,f)


#RandomForest_Train('./RF_train/features_RF_Cerrado.csv','./RF_train/labels_RF_Cerrado.csv','./RF_train/rf_model.cpickle')

def randomForest_Predict(root_add, model_file):

	with open(model_file,'rb') as f:
		rf = pickle.load(f)

	regions = ['Q3','Q4','Q5','Q6','Q7','Q8','Q9','Q10']
	file_type = ['Slope.tif', 'S1_Ann.tif', 'S1_AnnV.tif', 'S1_Dry.tif','S1_Wet.tif','S2_Ann.tif','S2_AnnV.tif','S2_Dry.tif','S2_Wet.tif']
	template_add = string.Template(root_add+'${region}/${region}_')

	#preds = rf.predict(new_X)

	for j in regions:
		print(j)
		img_slope = load_img(template_add.substitute(region=j)+file_type[0])
		img_s1_ann = load_img(template_add.substitute(region=j)+file_type[1])
		img_s1_annV = load_img(template_add.substitute(region=j)+file_type[2])
		img_s1_dry = load_img(template_add.substitute(region=j)+file_type[3])
		img_s1_wet = load_img(template_add.substitute(region=j)+file_type[4])
		img_s2_ann = load_img(template_add.substitute(region=j)+file_type[5])
		img_s2_annV = load_img(template_add.substitute(region=j)+file_type[6])
		img_s2_dry = load_img(template_add.substitute(region=j)+file_type[7])
		img_s2_wet = load_img(template_add.substitute(region=j)+file_type[8])

		result = np.zeros_like(img_slope)
		print(img_slope.shape)

		points = np.indices(img_slope.shape).reshape(2, -1).T
		print(points.shape)
		print(points)
		step=800
		for i in tqdm(range(0,result.shape[0],step)):
			print(i)
			if i+step<result.shape[0]:
				points_line=points[i*(result.shape[1]):(i+step)*result.shape[1],:]
				print(points_line)
				print('-------')
				result[i:i+step,:] = rf.predict(createFeatureSpace(points_line, img_s2_wet, img_s2_dry,
												img_s2_ann, img_s2_annV, img_s1_wet,
												img_s1_dry, img_s1_ann, img_s1_annV, img_slope)).reshape((step,result.shape[1]))
			else:
				size = result.shape[0]-i
				points_line=points[i*(result.shape[1]):result.shape[0]*result.shape[1],:]
				print(points_line)
				print('-------')
				result[i:result.shape[0],:] = rf.predict(createFeatureSpace(points_line, img_s2_wet, img_s2_dry,
												img_s2_ann, img_s2_annV, img_s1_wet,
												img_s1_dry, img_s1_ann, img_s1_annV, img_slope)).reshape((size,result.shape[1]))
			

		georrefData(result,template_add.substitute(region=j)+'CLS.tif',template_add.substitute(region=j)+file_type[0])

#randomForest_Predict('./RF_test/', './RF_train/rf_model.cpickle')