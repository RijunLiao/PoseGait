import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import pickle
import random
import cv2
import os
import scipy.io as sio
import json


def pose_loader(dir,path,frame = 17):
	#print("pose_loader work")
	#print(path)
	#pos1=path.rfind('/')
	#pos2=path.rfind('.')
	#impath=path[pos1+1:pos2]
	pose_dir = dir+path+"/"
	#im_path = pose_dir + impath + '.0001.mat'
	#print(im_path)
	#im_path = path
	image = np.zeros((126,frame),dtype='float')

	if os.path.exists(pose_dir):
		dirs = os.listdir(pose_dir)
		len_frames = len(dirs)
		#print(dirs)
		if len_frames == 0:
			print("lack")
		elif len_frames <= frame:
			for i in range(len_frames):
				pose_json = dirs[i]
				jsonFile = pose_dir + pose_json
				with open(jsonFile,'r') as f:
					json_dict = json.load(f)
					if(len(json_dict['people']) > 0):
						keypoints = json_dict['people'][0]['pose_keypoints_2d']
						keypoints_X = []
						keypoints_Y = []
						keypoints_C =[]

						mid_hipX = (keypoints[24] + keypoints[33])/2
						mid_hipY = (keypoints[25] + keypoints[34])/2
						neckX = keypoints[3]
						neckY = keypoints[4]
						H_square = pow((mid_hipX - neckX),2) + pow((mid_hipY - neckY),2)
						H_body = pow(H_square,0.5)
						if(H_body == 0):
							for j in range(18):
							
								keypoints_X.append(0)
								keypoints_Y.append(0)
								keypoints_C.append(0)
								#print("exception")
						else:
							for j in range(18):
								if(keypoints[j*3] == 0):
									keypoints_X.append(0)
									keypoints_Y.append(0)
									keypoints_C.append(0)
								else:
									keypoints_X.append((keypoints[j*3]-neckX)/H_body)
									keypoints_Y.append((keypoints[j*3+1]-neckY)/H_body)
									keypoints_C.append(keypoints[j*3+2])

				
				image[0:18,i] =  keypoints_X
				image[42:60,i] = keypoints_Y
				#image[84:102,i] = keypoints_C
			


		elif len_frames > frame:
		#	print(">")

			count = 0
			rand = random.randint(17,len_frames)
			for i in range(rand-17,rand):
				
				pose_json = dirs[i]
				#print(pose_json)

				jsonFile = pose_dir + pose_json
				with open(jsonFile,'r') as f:
					json_dict = json.load(f)
					
					if(len(json_dict['people']) > 0):
						keypoints = json_dict['people'][0]['pose_keypoints_2d']

						keypoints_X = []
						keypoints_Y = []
						keypoints_C =[]

						mid_hipX = (keypoints[24] + keypoints[33])/2
						mid_hipY = (keypoints[25] + keypoints[34])/2
						neckX = keypoints[3]
						neckY = keypoints[4]
						H_square = pow((mid_hipX - neckX),2) + pow((mid_hipY - neckY),2)
						H_body = pow(H_square,0.5)
						if(H_body == 0):
							for j in range(18):
							
								keypoints_X.append(0)
								keypoints_Y.append(0)
								keypoints_C.append(0)
								#print("exception")
						else:
							for j in range(18):
								if(keypoints[j*3] == 0):
									keypoints_X.append(0)
									keypoints_Y.append(0)
									keypoints_C.append(0)
								else:
									keypoints_X.append((keypoints[j*3]-neckX)/H_body)
									keypoints_Y.append((keypoints[j*3+1]-neckY)/H_body)
									keypoints_C.append(keypoints[j*3+2])

						
						image[0:18,count] =  keypoints_X
						image[42:60,count] = keypoints_Y
						#image[84:102,count] = keypoints_C
						count = count + 1
			#image=np.array(image)
			#image=torch.from_numpy(image)
			#image = image.type(torch.FloatTensor)

				

					
						
	#print(image)
	#image=torch.from_numpy(image)	
	#image = image.type(torch.FloatTensor)
	#image=image.permute(1,0).contiguous()
	#image = image.view(frame, 14, 9)
	return image

class BPEI_Dataset(Dataset):
	def __init__(self, txt, data_dir, transform=None, loader=pose_loader):
		#print("class BPEI_Dataset")
		fh = open(txt, 'r')
		imgs = []	
		for line in fh:
			line = line.strip('\n')
			
			line = line.rstrip()
			words = line.split()

			#print(impath)
			imgs.append((words[0],int(words[1])))
			

		self.imgs = imgs
		self.data_dir = data_dir
		self.loader = loader
		self.transform = transform
	

	def __getitem__(self, index):
		fpath, label = self.imgs[index]

		img = self.loader(self.data_dir,fpath)
		#print(img.size)
		if self.transform is not None:
			img = self.transform(img)		
		return img, label, fpath

	def __len__(self):
		return len(self.imgs)