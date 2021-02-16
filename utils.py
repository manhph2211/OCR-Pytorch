import numpy as np
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
	CLASS_NUM=len(lbe.classes_)
	new_targets=[lbe.transform(new_target) for new_target in new_targets]
	return new_targets,CLASS_NUM



def remove_duplicates(x):
    if len(x) < 2:
        return x
    fin = ""
    for j in x:
        if fin == "":
            fin = j
        else:
            if j == fin[-1]:
                continue
            else:
                fin = fin + j
    return fin


def decode_predictions(preds, encoder):
    preds = preds.permute(1, 0, 2)
    preds = torch.softmax(preds, 2)
    preds = torch.argmax(preds, 2)
    preds = preds.detach().cpu().numpy()
    cap_preds = []
    for j in range(preds.shape[0]):
        temp = []
        for k in preds[j, :]:
            k = k - 1
            if k == -1:
                temp.append("§")
            else:
                p = encoder.inverse_transform([k])[0]
                temp.append(p)
        tp = "".join(temp).replace("§", "")
        cap_preds.append(remove_duplicates(tp))
    return cap_preds



def split_data(img_paths,targets):
	X_train, X_test, y_train, y_test = model_selection.train_test_split(img_paths, targets, test_size=0.2, random_state=1)
	X_train, X_val, y_train, y_val = model_selection.train_test_split(X_train, y_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2
	return X_train,y_train,X_val,y_val,X_test,y_test


