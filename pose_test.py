import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
#import net
import networks
import resnet
import argparse
import BPEI_loader
from BPEI_loader import BPEI_Dataset
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "9"
import numpy as np
import torch.nn as nn
 
parser = argparse.ArgumentParser(description='Train')
parser.add_argument('-d', '--data_dir', type=str, default='dataset_pose/contour_bbx_images_all_part_pose_3D_add_feature/',
                    help="data dir")
parser.add_argument('--model', type=str, default='silh9-model/pose-200.pkl',
                    help="path to model")		
parser.add_argument('--test_list', type=str, default='list/gei_train.txt',
                    help="path to data list")						
parser.add_argument('-j', '--workers', default=1, type=int,
                    help="number of data loading workers (default: 4)")			
parser.add_argument('--train_batch', default=256, type=int,
                    help="train batch size")		
parser.add_argument('--test_batch', default=1, type=int,
                    help="train batch size")
parser.add_argument('--save_path', type=str, default='feature_pose/',
                    help="save path")						
args = parser.parse_args()					

		
test_data=BPEI_Dataset(args.test_list, args.data_dir, loader=BPEI_loader.posemulti_loader)	
test_loader = DataLoader(
    dataset=test_data,
	batch_size=args.test_batch, 
	shuffle=False, 
	num_workers=args.workers,
	pin_memory=True, 
	drop_last=True,
)

#device_ids = [0, 7, 9]
model = resnet.ResNet18(input_nc=1)
#model = nn.DataParallel(model,device_ids=device_ids).cuda()
model = nn.DataParallel(model).cuda()	
model.load_state_dict(torch.load(args.model))

#print(model.state_dict().keys())


#-------weight--------------------------------
# for i, j in model.named_parameters():
    # if i.find('gamma')>0:
		# print(i)
		# print(j)


save_path = args.save_path
if not os.path.exists(save_path):
	  os.makedirs(save_path)
for imgs, ids, fpath in test_loader:
	imgs = imgs[0]
	#imgs = imgs[0:16]
	#print(imgs.shape)
	imgs, ids = Variable(imgs), Variable(ids)
	imgs.data = imgs.data.type(torch.FloatTensor)
	imgs, ids = imgs.cuda(), ids.cuda()
	#print(len(fpath))
	fc,out = model(imgs)
	#fc,out = model(imgs, phase= 'TEST')
	#print(fc.data.shape)
	
	pos1=fpath[0].rfind('/')
	pos2=fpath[0].rfind('.')
	imtxt=fpath[0][pos1+1:pos2]	
	#print("file:")
	print(imtxt)
	feature_name=save_path+imtxt+'.txt'
	fout=open(feature_name, 'w+')
	fc=fc.cpu()
	ip1=fc[0].data
	#print(ip1.type)
	ip1=ip1.numpy()
	#print(len(ip1))
	for idx in xrange(len(ip1)):
		fout.write(str(ip1[idx])+' ')      
	fout.close()










