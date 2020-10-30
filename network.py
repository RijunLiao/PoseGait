import torch
import torch.nn as nn
import math
from torch.autograd import Variable
import torch.nn.functional as F

def conv7x7(in_planes, out_planes, stride=1, padding=1, group=1, bn=False):
	"""3x3 convolution with padding"""
	layers = []
	conv = nn.Conv2d(in_planes, out_planes, kernel_size=7, stride=stride,
					 padding=padding, groups=group, bias=False)
	layers.append(conv)
	if bn:
		bn_layer = nn.BatchNorm2d(out_planes)
		layers.append(bn_layer)
	layers.append(nn.ReLU(inplace=True))
	return nn.Sequential(*layers)

def conv3x3(in_planes, out_planes, stride=1, padding=1, group=1, bn=False):
	"""3x3 convolution with padding"""
	layers = []
	conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
					 padding=padding, groups=group, bias=False)
	layers.append(conv)
	if bn:
		bn_layer = nn.BatchNorm2d(out_planes)
		layers.append(bn_layer)
	layers.append(nn.ReLU(inplace=True))
	return nn.Sequential(*layers)
		
def conv1x1(in_planes, out_planes, stride=1, padding=1, group=1, bn=False):
	"""3x3 convolution with padding"""
	layers = []
	conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
					 padding=padding, groups=group, bias=False)
	layers.append(conv)

	if bn:
		bn_layer = nn.BatchNorm2d(out_planes)
		layers.append(bn_layer)
	layers.append(nn.ReLU(inplace=True))
	return nn.Sequential(*layers)


class RecSilhAttnGNet(nn.Module):
	"""
	input_size(3,128,128)
	"""

	def __init__(self, input_nc):
		super(RecSilhAttnGNet, self).__init__()
		self.bn = True
		
		#self.attn1 = X_Attn(1,nn.ReLU(inplace=True))
		self.conv1 = conv3x3(input_nc, 32, stride=1, padding=0, group=1, bn=self.bn)
		self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
		#self.conv1_1 = conv1x1(16, 16, stride=1, padding=0, group=1, bn=self.bn)
		#self.attn = FrameAttn(16, nn.ReLU(inplace=True))

		# self.block1_attn = SelfAttn(32, nn.ReLU(inplace=True))
		self.conv2 = conv3x3(32, 64, stride=1, padding=0,  bn=self.bn)
		self.conv2_1 = conv3x3(64, 64, stride=1, padding=1, group=1, bn=self.bn)
		#self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
		# self.block2_attn = SelfAttn(32, nn.ReLU(inplace=True))
		self.conv3 = conv3x3(64, 128, stride=1, padding=0, group=1, bn=self.bn)
		#self.conv3_1 = conv1x1(64, 64, stride=1, padding=0, group=1, bn=self.bn)

		self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)



		self.conv4 = conv3x3(128, 128, stride=1, padding=1, group=1, bn=self.bn)
		self.conv4_1 = conv3x3(128,128, stride=1, padding=1, group=1, bn=self.bn)

		self.drop = nn.Dropout(p=0.5)
	
		self.fc = nn.Linear(128*29*2, 512)  # conv nn.Linear(256 * 9 * 9 * 2, 2) pool nn.Linear(256 * 10 * 10, 128)

		self.fc_cls = nn.Linear(512, 5)
		
		# for p in self.parameters():
			# p.requires_grad=True
			
		#self.W = nn.Parameter(torch.zeros(8*27*17, 8*27*17))
		#self.bn1 = nn.BatchNorm1d(88)
		#self.conv1x1 = conv1x1(64, 8, stride=1, padding=0, group=1, bn=self.bn)

	def forward(self, x, phrase = 'TRAIN'):
	
		feature=[]
		
		batch, pose,t = x.size()
		
		x = x.view(batch, 1, pose,t)

		feature.append(x.data)
		
		x = self.conv1(x)
	
		x = self.conv2(x)
		
		pool1 = self.pool1(x)
		
		x = self.conv2_1(pool1)

	
		x = self.conv2_1(x)
		x = pool1 + x
		
		x = self.conv3(x)
	
		pool2 = self.pool2(x)
	
		x = self.conv4(pool2)
		
		x = self.conv4_1(x)
		
		out = pool2 + x
		
		#fc = out
		#feature = out
		
		x = out.view(out.size(0), -1)
		
	
		if(phrase == 'TRAIN'):
			x = self.drop(x)
		
		fc = self.fc(x)
		
		out = self.fc_cls(fc)
		


		#x, attn = self.attn(x)
		#batch, C, t, height, width = x.size()
		#x = x.permute(0,2,1,3,4).view(batch*t, C, height, width)
		'''
		x = self.pool1(x)
		print(x.size())
		#x = self.conv1_1(x)

		x = self.conv2(x)
		x = self.pool2(x)
		x = self.conv2_1(x)
		#print(x.size())		
		#gamma = self.bn1(self.W)
		# gamma= F.softmax(gamma,dim=-1)
		#print(gamma.size())
		gamma = gamma.unsqueeze(0).repeat(batch,1,1)
		x = self.conv1x1(x)

		x = x.view(batch, t, -1)
		#wx = x.mul(gamma)
		wx = torch.bmm(x,gamma)
		x = wx
		x = x.view(batch*t, 1, height, width)
		#wx = wx.view(batch*t, 1, height, width)
		
		x = self.conv3(x)
		x = self.conv3_1(x)
		# x = self.pool3(x)
		x = self.conv4(x)
		x = self.conv4_1(x)
		#gamma = torch.Tensor([2, 0.5, 0.5, 0.5, 0.5, 3]).cuda()
		#x = putWeight(x, gamma)	
		#print(x.size())
		x = x.view(x.size(0), -1)
		if(phrase == 'TRAIN'):
			x = self.drop(x)
		fc = self.fc(x)
		fc = ave_feature(fc, t)
		if(phrase == 'TEST'):
			fc = ave_feature(fc, fc.size(0))  # batch/n , 256

		out = self.fc_cls(fc)
'''
		return fc, out