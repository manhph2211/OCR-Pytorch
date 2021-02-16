import os
import re
import cv2
import matplotlib.pyplot as plt 
from sklearn import preprocessing
from sklearn import model_selection


def get_data(root_path='./captcha_images_v2'):
	labels=os.listdir(root_path)
	targets=[re.sub('.png$','',x) for x in labels]
	img_paths=[os.path.join(root_path,x) for x in labels]
	return img_paths,targets


def show_img(path):
	img=cv2.imread(path)
	plt.imshow(img)
	plt.show()


def encode_targets(targets):
	new_targets=[]
	for target in targets:
		new_target=[cha for cha in target]
		new_targets.append(new_target)
	# encoding
	flatten_targets=[x for y in new_targets for x in y]
	lbe=preprocessing.LabelEncoder()
	lbe.fit(flatten_targets)
	#CLASS_NUM=len(lbe.classes_)
	new_targets=[lbe.transform(new_target) for new_target in new_targets]
	return new_targets,lbe


def split_data(img_paths,targets):
	X_train, X_test, y_train, y_test = model_selection.train_test_split(img_paths, targets, test_size=0.2, random_state=1)
	X_train, X_val, y_train, y_val = model_selection.train_test_split(X_train, y_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2
	return X_train,y_train,X_val,y_val,X_test,y_test


