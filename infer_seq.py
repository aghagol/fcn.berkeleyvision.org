import numpy as np
from PIL import Image
import sys, os
sys.path.append('/home/mo/caffe/python')
os.environ['GLOG_minloglevel'] = '2'
import caffe
import cv2
import matplotlib.pyplot as plt
import time
caffe.set_mode_gpu()

batch_size = 10
net = caffe.Net('voc-fcn16s/deploy.prototxt', 'voc-fcn16s/fcn16s-heavy-pascal.caffemodel', caffe.TEST)

data_path = '/home/mo/Desktop/ROS_data/jaguar/2016-07-22/'
image_list_file = data_path+'dataset_train_nn.txt'
image_list = np.loadtxt(image_list_file)[:,0]
for counter, image_name in enumerate(image_list):
	print 'working on image %f'%image_name
	im = cv2.imread(data_path+'train/left/%f.png'%image_name)
	in_ = np.array(im, dtype=np.float32)
	# in_ = in_[:,:,::-1]
	in_ -= np.array((104.00698793,116.66876762,122.67891434))
	in_ = in_.transpose((2,0,1))
	
	# shape for input (data blob is N x C x H x W), set data
	if not counter%batch_size:
		if counter==0:
			net.blobs['data'].reshape(batch_size, *in_.shape)
		else:
			net.forward()
	net.blobs['data'].data[counter%batch_size,...] = in_	
	if not counter%batch_size and counter>0:
		for sample in range(batch_size):
			out = net.blobs['score'].data[sample].argmax(axis=0)
			plt.imshow(im_list[sample][1])
			plt.imshow(out,alpha=.5)
			plt.savefig('out/%f.png'%im_list[sample][0])
			plt.clf()
	if counter%batch_size:
		im_list.append((image_name,np.array(im, dtype=np.float32)))
	else:
		im_list = []
		im_list.append((image_name,np.array(im, dtype=np.float32)))
