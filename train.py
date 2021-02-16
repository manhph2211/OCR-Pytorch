import torch
import torch.nn as nn
from utils import get_data, encode_targets, split_data,remove_duplicates,decode_predictions
from dataloader import make_dataset
from model import CaptchaModel
import engine
import numpy as np 


def train(BATCH_SIZE,NUM_EPOCH,NUM_WORKERS=8):

	img_paths,targets=get_data()
	encode_targets_,class_num=encode_targets(targets)
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

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	model=CaptchaModel(num_chars=class_num)
	model.to(device)	
	optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=5, verbose=True)
	loss_fn = nn.CrossEntropyLoss()

	for epoch in range(NUM_EPOCH):
		train_loss = engine.train_fn(model, train_loader, optimizer,device,loss_fn)
		valid_preds, test_loss = engine.eval_fn(model, val_loader,device,loss_fn)
		valid_captcha_preds = []
		for vp in valid_preds:
			current_preds = decode_predictions(vp, lbl_enc)
			valid_captcha_preds.extend(current_preds)    	
		combined = list(zip(test_targets_orig, valid_captcha_preds))
		print(combined[:10])
		test_dup_rem = [remove_duplicates(c) for c in test_targets_orig]
		accuracy = metrics.accuracy_score(test_dup_rem, valid_captcha_preds)  
		print(f"Epoch={epoch}, Train Loss={train_loss}, Test Loss={test_loss} Accuracy={accuracy}")
		scheduler.step(test_loss)

if __name__ == "__main__":
    train(8,10)