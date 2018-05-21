import numpy as np
import random as rd
import time
import os
import math
import scipy.ndimage.morphology as sci_morph
import scipy.ndimage.measurements as sci_measure
from PIL import Image as PImage
import cv2

def extractObj(obj_nb,aloi_db_type,mask_thresh_flag):

	### MACROS
	D_E_B_U_G = 0

	pwd = os.getcwd()
	ALOI_SRC = '/media/bruno/Storage Disk/INICIACAO - DORIS/aloi'
	ALOI_ILL_SRC = '/media/bruno/Storage Disk/INICIACAO - DORIS/aloi/aloi_ill'
	ALOI_ROT_SRC = '/media/bruno/Storage Disk/INICIACAO - DORIS/aloi/aloi_rot'
	ALOI_MASK_SRC = '/media/bruno/Storage Disk/INICIACAO - DORIS/aloi/aloi_mask'
	if not (os.path.exists(ALOI_SRC)):
		ALOI_SRC = '/home/bruno.afonso/datasets/aloi'
		ALOI_ILL_SRC = '/home/bruno.afonso/datasets/aloi/aloi_ill'
		ALOI_ROT_SRC = '/home/bruno.afonso/datasets/aloi/aloi_rot'
		ALOI_MASK_SRC = '/home/bruno.afonso/datasets/aloi/aloi_mask'
	### END OF MACROS


	# READ A RANDOM FRAME FROM A SELECTED OBJECT IN ALOI_ILL
	if (aloi_db_type == 0):
		temp_pth = os.path.join(ALOI_ILL_SRC,str(obj_nb))
	elif (aloi_db_type == 1):
		temp_pth = os.path.join(ALOI_ROT_SRC,str(obj_nb))

	os.chdir(temp_pth)
	lst_dir_obj = os.listdir(temp_pth)
	sz_lst_dir = len(lst_dir_obj)

	flag_ok_frame = False
	while flag_ok_frame != True:
		rnd_obj_fr = rd.randint(0,sz_lst_dir-1)
		base_str = lst_dir_obj[rnd_obj_fr]
		if base_str.find('r0') < 0:
			flag_ok_frame = True

	img_obj = cv2.imread(lst_dir_obj[rnd_obj_fr])

	# DETECT POSITION NAME OF THE OBJECT FRAME
	if (aloi_db_type == 0):
		if 'c1' in lst_dir_obj[rnd_obj_fr]:
			pos_str_obj = 'c1'
		elif 'c2' in lst_dir_obj[rnd_obj_fr]:
			pos_str_obj = 'c2'
		elif 'c3' in lst_dir_obj[rnd_obj_fr]:
			pos_str_obj = 'c3'
	elif (aloi_db_type == 1):
		base_str = lst_dir_obj[rnd_obj_fr]
		idx_b = base_str.find('_') + 1
		idx_f = base_str.find('.')
		pos_str_obj = base_str[idx_b:idx_f]

	# READ THE MASK IMAGE FOR THAT SAME FRAME
	temp_pth = os.path.join(ALOI_MASK_SRC,str(obj_nb))
	os.chdir(temp_pth)
	img_mask = cv2.imread(str(obj_nb) + '_' + pos_str_obj + '.png')

	# EXTRACT OBJECT FROM ORIGINAL FRAME
	os.chdir(pwd)
	print(obj_nb)
	print(img_obj.shape)
	print(img_mask.shape)
	img_final = np.multiply(img_obj,img_mask/255)

	if (D_E_B_U_G):
		cv2.imwrite('test_final_extract.png',img_final)

	if (mask_thresh_flag == 0):
		return img_final

	img_final_mask = np.array(img_final)
	img_final_mask = img_final_mask.astype(np.uint8)
	img_final_mask = cv2.cvtColor(img_final_mask, cv2.COLOR_RGB2GRAY)
	#th_val, img_final_mask = cv2.threshold(img_final_mask,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	th_val, img_final_mask = cv2.threshold(img_final_mask,25,255,cv2.THRESH_BINARY)
	img_final_mask = np.expand_dims(img_final_mask,axis=2)
	img_final_mask = np.concatenate((img_final_mask, img_final_mask, img_final_mask),axis=2)

	img_new_final = np.multiply(img_obj,img_final_mask/255)

	if (D_E_B_U_G):
		cv2.imwrite('test_final_extract_new.png',img_new_final)

	if (mask_thresh_flag == 1):
		return img_new_final

	return img_final

'''
#asdf = rd.randint(1,1000)
asdf = 854
#print(asdf)
extractObj(asdf,1,0)
'''