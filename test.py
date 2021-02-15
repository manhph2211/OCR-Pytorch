import cv2
import easyocr
from utils import show_img,get_data


def predict(img_paths):
	reader=easyocr.Reader(['en'],gpu=False)
	results=[reader.readtext(img_path) for img_path in img_paths]
	return results


def evaluate(results,targets):
	test=results-targets
	print(test.count(0))


img_paths,targets=get_data()
results=predict(img_paths[0:3])
evaluate(results,targets[0:3])

