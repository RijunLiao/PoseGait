import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
#import net
import network
import resnet
import argparse
import Loader
from Loader import BPEI_Dataset
#from centerloss import CenterLoss
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "9"
import torch.nn as nn
 
parser = argparse.ArgumentParser(description='Train')
parser.add_argument('-d', '--data_dir', type=str, default='train62/',
					help="data dir")
parser.add_argument('--train_list', type=str, default='list/train62.txt',
					help="path to data list")						
parser.add_argument('-j', '--workers', default=4, type=int,
					help="number of data loading workers (default: 4)")			
parser.add_argument('--train_batch', default=8, type=int,
					help="train batch size")		
parser.add_argument('--test_batch', default=32, type=int,
					help="train batch size")						
parser.add_argument('--max_epoch', default=200, type=int,
					help="maximum epochs to run")
parser.add_argument('--snapshot', default=50, type=int,
					help="maximum epochs to run")
parser.add_argument('--lr', '--learning-rate', default=0.000001, type=float,
					help="initial learning rate, use 0.0001 for rnn, use 0.0003 for pooling and attention")
args = parser.parse_args()					

train_data=BPEI_Dataset(args.train_list, args.data_dir, loader=Loader.pose_loader)		
train_loader = DataLoader(
	dataset=train_data,
	batch_size=args.train_batch, 
	shuffle=True, 
	num_workers=args.workers,
	pin_memory=True, 
	drop_last=True,
)
	
#device_ids = [0, 1, 2, 3]	
model = network.RecSilhAttnGNet(input_nc = 1)
model = nn.DataParallel(model).cuda()
print(model)
#model = nn.DataParallel(model, device_ids=device_ids).cuda()

#model.load_state_dict(torch.load('silh9-model/pose-200.pkl'))
#model.load_state_dict(torch.load('silh9_attn_model/cb_ave-200.pkl'))
#print(model)

#loss_weight = 1
loss_func = torch.nn.CrossEntropyLoss()
#centerloss = CenterLoss(74, 256)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
#optimcenter = torch.optim.Adam(centerloss.parameters(),lr = 5*args.lr)


for epoch in range(args.max_epoch):
#for epoch in range(5):
	print('epoch {}'.format(epoch + 1))
	# training-----------------------------
	train_loss = 0.
	cen_loss = 0.
	train_acc = 0.
	for imgs, ids, fpath in train_loader:
		imgs, ids = Variable(imgs), Variable(ids)
  		#print(imgs.type
		#imgs=imgs.byte()
		imgs.data = imgs.data.type(torch.FloatTensor)
		ids.data = ids.data.type(torch.LongTensor)
		imgs, ids = imgs.cuda(), ids.cuda()
	
		fc,out = model(imgs)
	
		#out = out.float
		loss = loss_func(out, ids)

		
		train_loss += loss.item()
		pred = torch.max(out, 1)[1]
		train_correct = (pred == ids).sum()
		train_acc += train_correct.item()
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		#optimizer.module.step()
	print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (len(train_data)), train_acc / (len(train_data))))
	#print('center Loss: {:.6f}'.format(cen_loss / (len(train_data))))
	if epoch%args.snapshot==0:
		model_path='model/pose-'+str(epoch)+'.pkl'
		torch.save(model.state_dict(), model_path)
		
model_path='model/pose-'+str(args.max_epoch)+'.pkl'
torch.save(model.state_dict(), model_path)










