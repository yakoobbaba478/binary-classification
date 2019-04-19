import os
import sys
import numpy as np
import cv2
from os import listdir
from os.path import isfile,isdir,join,exists
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim


class Net(nn.Module):

	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(1, 6, 5)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.fc1 = nn.Linear(16 * 5 * 5, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)
		#nn.Softmax())
	def forward(self, x):
		x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
		x = F.max_pool2d(F.relu(self.conv2(x)), 2)
		x = F.interpolate(x, (5, 5), mode = 'bilinear')
		x = x.view(x.size(0), 5*5*16)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.softmax(self.fc3(x))
		#x = F.log_softmax(x, dim=1)
		return x
	def num_flat_features(self, x):
		size = x.size()[1:]
		num_features = 1
		for s in size:
			num_features *= s
		return num_features

def train(src_path):

	global net

	params = list(net.parameters())
	print(len(params))
	print(params[0].size())

	for class_dir in listdir(src_path):
		print ("scanning dir :",class_dir)
		if(isfile(class_dir)):
			continue
		for img_k in listdir(join(src_path,class_dir)):
			print ("read image:",join(src_path,class_dir,img_k))		
			img_gray = cv2.imread(os.path.join(src_path,class_dir,img_k),0)
			
			img_resize = cv2.resize(img_gray,(28, 28))

			optimizer = optim.SGD(net.parameters(), lr=0.01)
			np_img = np.array([[img_resize]])
			np_img = np.divide(np_img, 255.0)
			np_img = torch.from_numpy(np_img).double()	
			print(np_img.size)
			optimizer.zero_grad()
			print (np_img.shape)
			output = net(np_img.float())
			print ("output:[",class_dir,"]",output)	
			print (torch.max(output, 1))
			target = np.zeros(10)
			target[int(class_dir)] = 1 
			target = torch.from_numpy(target).float()
			target = target.view(1, -1)
			criterion = nn.MSELoss() 
			loss = criterion(output, target)
			print ("losss:",loss)
			loss.backward()
			optimizer.step()
	

def predict(img_path):
	global net
	img_gray = cv2.imread(os.path.join(img_path),0)
	img_resize = cv2.resize(img_gray,(28, 28))
	for row in range(28):
		for col in range(28):
			if (img_resize[row][col] == 255):
				img_resize[row][col] = 0
			elif (img_resize[row][col] == 0):
				img_resize[row][col] = 255
	cv2.imshow("1",img_resize)
	cv2.waitKey()			

	np_img = np.array([[img_resize]])
	np_img = np.divide(np_img, 255.0)
	np_img = torch.from_numpy(np_img).double()	
	print(np_img.size)
	output = net(np_img.float())
	#print (torch.max(output, 1))
	print ("the output is: ",output)

if __name__=="__main__":
	global net
	net = Net()
	mode = sys.argv[1]
	if(mode == "train"):
		src_path = sys.argv[2]
		for i in range(40):
			train(src_path)
		val_img = sys.argv[3]
		predict(val_img)

	elif(mode == "eval"):
		img_path = sys.argv[2]
		model_path = sys.argv[3]
		net.load_state_dict(torch.load(model_path))
		net.eval()	
		predict(img_path)
		
	torch.save(net.state_dict(), "/home/baba/saveddata.pth")
