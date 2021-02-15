import cv2
import easyocr
from utils import show_img,get_data
import numpy as np


def predict(img_paths):
	reader=easyocr.Reader(['en'],gpu=False)
	results=[reader.readtext(img_path) for img_path in img_paths]
	return results


def evaluate(results,targets):
	print(len(np.where(results==targets))/len(results))


img_paths,targets=get_data()
results=predict(img_paths[1:5])
evaluate(results,targets[1:5])

