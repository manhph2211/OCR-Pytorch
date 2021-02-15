import numpy as np
import os
import re

def get_data(root_path='./captcha_images_v2'):
	labels=os.listdir(root_path)
	targets=[re.sub('.png$','',x) for x in labels]
	img_paths=[os.path.join(root_path,x) for x in labels]
	return img_paths,targets


