import torch
import cv2
from torchvision import transforms


class make_dataset:

	def __init__(self,img_paths,targets,resize=True):
		self.img_paths=img_paths
		self.targets=targets
		self.resize=resize
		self.transforms=transforms.Compose([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


	def __len__(self):
		return len(self.img_paths)

	def __getitem__(self,idx):

		img=cv2.imread(self.img_paths[idx])

		if self.resize:
			img=cv2.resize(img,(300,75))
		img=torch.tensor(img, dtype=torch.float32)
		img=img.permute(2, 0, 1)
		img=self.transforms(img)

		tar=self.targets[idx]
		tar=torch.tensor(tar,dtype=torch.long)

		return img,tar 

