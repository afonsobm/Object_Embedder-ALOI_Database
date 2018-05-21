import numpy as np
import random as rd
import time
import os
import math
import re
import scipy.ndimage.morphology as sci_morph
import scipy.ndimage.measurements as sci_measure
from PIL import Image as PImage
import cv2
import Rnd_Obj_Extract

def chooseRndBG():

	
	pwd = os.getcwd()

	VDAO_FRAMES_SRC = '/media/bruno/Storage Disk/INICIACAO - DORIS/vdao/Reference_Object_frames_skip_17_full_aligned'
	nb_frames_offset = 2
	if not (os.path.exists(VDAO_FRAMES_SRC)):
		VDAO_FRAMES_SRC = '/local/home/common/datasets/Reference_Object_frames_skip_17_full_aligned'
		nb_frames_offset = 1

	VDAO_DATABASE_LIGHTING_ENTRIES = np.array(['NORMAL-Light','EXTRA-Light']) 

	VDAO_DATABASE_OBJECT_ENTRIES = np.array(['Black_Backpack',
											 'Black_Coat',
											 'Brown_Box',
											 'Camera_Box',
											 'Dark-Blue_Box',
											 'Pink_Bottle',
											 'Shoe',
											 'Towel',
											 'White_Jar'])

	VDAO_DATABASE_OBJECT_POSITION_ENTRIES = np.array(['POS1','POS2','POS3']) 

	VDAO_DATABASE_REFERENCE_ENTRIES = np.array(['Pink_Bottle;Towel;Black_Coat;Black_Backpack',
												'Shoe;Dark-blue_Box;Camera_Box',
												'White_Jar;Brown_Box'])


	# CHOOSES A RANDOM OBJECT AND POSITION TO SEACH FOR A OBJECTLESS BACKGROUND FRAME
	rnd_obj_chs = rd.randint(0,VDAO_DATABASE_OBJECT_ENTRIES.shape[0]-1)
	rnd_pos_chs = rd.randint(0,VDAO_DATABASE_OBJECT_POSITION_ENTRIES.shape[0]-1)
	temp_pth_str = VDAO_DATABASE_OBJECT_ENTRIES[rnd_obj_chs] + '_' + VDAO_DATABASE_OBJECT_POSITION_ENTRIES[rnd_pos_chs] + '_target'
	temp_pth_bg_tar = os.path.join(VDAO_FRAMES_SRC,VDAO_DATABASE_LIGHTING_ENTRIES[1],temp_pth_str)
	while not (os.path.exists(temp_pth_bg_tar)):
		rnd_pos_chs = rd.randint(0,VDAO_DATABASE_OBJECT_POSITION_ENTRIES.shape[0]-1)
		temp_pth_str = VDAO_DATABASE_OBJECT_ENTRIES[rnd_obj_chs] + '_' + VDAO_DATABASE_OBJECT_POSITION_ENTRIES[rnd_pos_chs] + '_target'
		temp_pth_bg_tar = os.path.join(VDAO_FRAMES_SRC,VDAO_DATABASE_LIGHTING_ENTRIES[1],temp_pth_str)
	temp_pth_str = VDAO_DATABASE_OBJECT_ENTRIES[rnd_obj_chs] + '_' + VDAO_DATABASE_OBJECT_POSITION_ENTRIES[rnd_pos_chs] + '_reference'
	temp_pth_bg_ref = os.path.join(VDAO_FRAMES_SRC,VDAO_DATABASE_LIGHTING_ENTRIES[1],temp_pth_str)

	# CREATES A LIST WITH THE NUMBERS OF ALL FRAMES WHICH CONTAIN AN OBJECT
	os.chdir(temp_pth_bg_tar)
	lst_dir_bg = os.listdir(temp_pth_bg_tar)
	sz_lst_dir_bg = len(lst_dir_bg) - nb_frames_offset
	with open('object_frames.txt','r') as f:
		content = f.readlines()
		content = [x.strip() for x in content]
		if '' in content:
			content = content[0:-1]
		if (len(content) == 2):
			obj_frames = np.array(list(map(int,re.findall('\d+',content[0]))))
			obj_frames = np.concatenate((obj_frames,np.array(list(map(int,re.findall('\d+',content[1]))))),axis=0)
			obj_frames = np.concatenate((np.arange(obj_frames[0],obj_frames[1]+1),np.arange(obj_frames[2],obj_frames[3]+1)), axis=0)
		else:
			obj_frames = np.array(list(map(int,re.findall('\d+',content[0]))))
			obj_frames = np.arange(obj_frames[0],obj_frames[1]+1)

	# CHOOSES A RANDOM OBJECTLESS BACKGROUND FRAME
	temp_bool = False
	while not (temp_bool):
		rnd_bg_fr = rd.randint(0,sz_lst_dir_bg-1)
		if not rnd_bg_fr in obj_frames:
			temp_bool = True

	img_ref_temp_tar = cv2.imread('frame_' + str(rnd_bg_fr) + '.png',3)

	os.chdir(temp_pth_bg_ref)
	img_ref_temp_ref = cv2.imread('frame_' + str(rnd_bg_fr) + '.png',3)

	os.chdir(pwd)
	return img_ref_temp_tar, img_ref_temp_ref



def blendObj(rnd_ext_obj_nb):

	### MACROS
	D_E_B_U_G = 1
	ALOI_DATASET_FLAG = 1
	ALOI_MASK_THRESHOLD_FLAG = 0
	OBJ_RESIZE_PERCENTAGE_THRESHOLD = 0.7
	REF_IMG_HEIGHT = 720
	REF_IMG_WIDTH = 1280
	#STREL_KERNEL = np.matrix('0,1,0;1,1,1;0,1,0')
	STREL_KERNEL = cv2.getStructuringElement(shape=1, ksize=(3,3))
	### END OF MACROS


	# LOAD OBJ AND REF IMAGES
	img_org = Rnd_Obj_Extract.extractObj(rnd_ext_obj_nb,ALOI_DATASET_FLAG,ALOI_MASK_THRESHOLD_FLAG)
	img_bg_tar, img_bg_ref = chooseRndBG()
	obj_rows, obj_cols, obj_channels = img_org.shape

	# RANDOM ROTATE OBJ IMAGE
	rnd_nb = rd.randint(0,360)
	M = cv2.getRotationMatrix2D((obj_cols/2,obj_rows/2),rnd_nb,1)
	img_rot = cv2.warpAffine(img_org,M,(obj_cols,obj_rows))

	if (D_E_B_U_G):
		cv2.imwrite('test_rot.png',img_rot)

	# RANDOM RESIZE OBJ IMAGE
	rnd_nb = 0
	while (rnd_nb < OBJ_RESIZE_PERCENTAGE_THRESHOLD):
		rnd_nb = rd.random()
	img_rsz = cv2.resize(img_rot, None, fx=rnd_nb, fy=rnd_nb, interpolation=cv2.INTER_AREA)
	rsz_rows, rsz_cols, rsz_channels = img_rsz.shape

	if (D_E_B_U_G):
		cv2.imwrite('test_rsz.png',img_rsz)

	# CREATE MASK FOR OBJECT PLACEMENT
	img_mask = np.zeros((REF_IMG_HEIGHT,REF_IMG_WIDTH,3))
	bool_conf = False

	while not (bool_conf):
		rnd_nb_row = rd.randint(0,REF_IMG_HEIGHT)
		rnd_nb_col = rd.randint(0,REF_IMG_WIDTH)

		if( ((rnd_nb_row + rsz_rows) < REF_IMG_HEIGHT) and ((rnd_nb_col + rsz_cols) < REF_IMG_WIDTH) ):
			bool_conf = True

	img_mask[rnd_nb_row:(rnd_nb_row + rsz_rows), rnd_nb_col:(rnd_nb_col + rsz_cols), 0:3] = img_rsz
	#np.savetxt('test.txt',img_mask, fmt='%f')

	if (D_E_B_U_G):
		cv2.imwrite('test_mask.png',img_mask)

	### OBJECT PLACEMENT ON TARGET FRAME ###
	
	# BINARIZATION OF THE MASK
	img_mask_bin = np.array(img_mask)
	img_mask_bin [ img_mask_bin > 0 ] = 255
	img_mask_bin = 255 - img_mask_bin

	# PERFORMS DILATATION OF THE MASK TO REMOVE BLACK BORDER
	img_mask_bin = img_mask_bin.astype(np.uint8)
	img_mask_bin = cv2.cvtColor(img_mask_bin, cv2.COLOR_RGB2GRAY)
	tresh_val, img_mask_bin = cv2.threshold(img_mask_bin, 127, 255, cv2.THRESH_BINARY)
	img_mask_bin_dilate = cv2.dilate(src=img_mask_bin, kernel=STREL_KERNEL, iterations=7)
	tresh_val, img_mask_bin_inv_dilate = cv2.threshold(img_mask_bin_dilate, 127, 255, cv2.THRESH_BINARY_INV)
	img_mask_bin_backg_dilate = cv2.dilate(src=img_mask_bin_dilate, kernel=STREL_KERNEL, iterations=4)
	img_mask_bin_backg_inv_dilate = cv2.bitwise_not(img_mask_bin_backg_dilate)
	img_mask_bin_inv_dilate_ORG = np.array(img_mask_bin_inv_dilate)


	# TRANSFORMS BINARY IMAGE OF THE DILATATED MASKS INTO RGB
	img_mask_bin_dilate = np.expand_dims(img_mask_bin_dilate,axis=2)
	img_mask_bin_dilate = np.concatenate((img_mask_bin_dilate, img_mask_bin_dilate, img_mask_bin_dilate),axis=2)
	img_mask_bin_inv_dilate = np.expand_dims(img_mask_bin_inv_dilate,axis=2)
	img_mask_bin_inv_dilate = np.concatenate((img_mask_bin_inv_dilate, img_mask_bin_inv_dilate, img_mask_bin_inv_dilate),axis=2)	
	img_mask_bin_backg_dilate = np.expand_dims(img_mask_bin_backg_dilate,axis=2)
	img_mask_bin_backg_dilate = np.concatenate((img_mask_bin_backg_dilate, img_mask_bin_backg_dilate, img_mask_bin_backg_dilate),axis=2)
	img_mask_bin_backg_inv_dilate = np.expand_dims(img_mask_bin_backg_inv_dilate,axis=2)
	img_mask_bin_backg_inv_dilate = np.concatenate((img_mask_bin_backg_inv_dilate, img_mask_bin_backg_inv_dilate, img_mask_bin_backg_inv_dilate),axis=2)

	# GENERATES NEW MASKS
	img_mask_new = np.multiply(img_mask,img_mask_bin_inv_dilate/255)
	img_mask_new_backg = np.multiply(img_bg_tar,img_mask_bin_backg_dilate/255)
	img_mask_bin_dif_dilate = np.subtract(img_mask_bin_inv_dilate,img_mask_bin_backg_inv_dilate)

	if D_E_B_U_G:
		cv2.imwrite('test_mask_1.png',img_mask_bin_inv_dilate)	
		cv2.imwrite('test_mask_2.png',img_mask_bin_backg_inv_dilate)	
		cv2.imwrite('test_mask_3.png',img_mask_bin_dif_dilate)
		cv2.imwrite('test_mask_notb_1.png',img_mask_new)	
		cv2.imwrite('test_mask_notb_2.png',img_mask_new_backg)	

	# GENERATE BLENDED IMAGE WITHOUT WEIGHTED SUM
	img_ref_cut = np.multiply(img_bg_tar,img_mask_bin_dilate/255)
	img_final = np.add(img_mask_new, img_ref_cut)

	# DISTANCE TRANSFORM FUNCTION FOR THE BORDER OF THE OBJECT
	img_dist_transf = cv2.distanceTransform(src=img_mask_bin_inv_dilate_ORG, distanceType=cv2.DIST_L2, maskSize=5)
	if D_E_B_U_G:
		cv2.imwrite('test_dist_transf.png',img_dist_transf)	

	img_dist_transf = np.expand_dims(img_dist_transf,axis=2)
	img_dist_transf = np.concatenate((img_dist_transf, img_dist_transf, img_dist_transf),axis=2)
	img_dist_transf_border = np.multiply(img_dist_transf, img_mask_bin_dif_dilate/255)
	for i in range(3):
		max_val = np.max(np.max(img_dist_transf_border[:,:,i]))
		img_dist_transf_border[:,:,i] = img_dist_transf_border[:,:,i] / max_val

	# GENERATING WEIGHT "IMAGE"
	img_weight_foreg = np.array(img_dist_transf_border)
	img_weight_foreg [img_mask_bin_backg_inv_dilate == 255] = 1
	img_weight_backg = 1 - np.array(img_weight_foreg)

	img_blur_border = np.add(np.multiply(img_weight_foreg,img_mask_new),np.multiply(img_weight_backg,img_mask_new_backg))

	if D_E_B_U_G:
		cv2.imwrite('test_final.png',img_blur_border)

	return img_blur_border, img_bg_ref, img_bg_tar

asdf = 315
blendObj(asdf)