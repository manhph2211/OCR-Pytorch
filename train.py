import torch
import torch.nn as nn
from utils import get_data, encode_targets, split_data
from dataloader import make_dataset
from model import CaptchaModel
import engine



def train(BATCH_SIZE,NUM_EPOCH,NUM_WORKERS=8):

	img_paths,targets=get_data()
	encode_targets_,lbe=encode_targets(targets)
	class_num=len(lbe.classes_)
	#print("class_num",class_num)
	X_train,y_train,X_val,y_val,X_test,y_test=split_data(img_paths,encode_targets_)

	train_dataset=make_dataset(X_train,y_train)
	train_loader = torch.utils.data.DataLoader(
	train_dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    shuffle=True,
    )

	val_dataset=make_dataset(X_val,y_val)
	val_loader = torch.utils.data.DataLoader(
	val_dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    shuffle=True,
    )

	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	model=CaptchaModel(num_chars=class_num)
	model.to(device)	
	optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=5, verbose=True)
	
	for epoch in range(NUM_EPOCH):
		train_loss = engine.train_fn(model, train_loader, optimizer,device)
		test_loss = engine.eval_fn(model, val_loader,device)
		print(f"Epoch={epoch}, Train Loss={train_loss}, Test Loss={test_loss} ")
		scheduler.step(test_loss)
if __name__ == "__main__":
    train(8,10)