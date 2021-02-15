import numpy as np
import os
import re
import cv2
import matplotlib.pyplot as plt 


def get_data(root_path='./captcha_images_v2'):
	labels=os.listdir(root_path)
	targets=[re.sub('.png$','',x) for x in labels]
	img_paths=[os.path.join(root_path,x) for x in labels]
	return img_paths,targets


def show_img(path):
	img=cv2.imread(path)
	plt.imshow(img)
	plt.show()


