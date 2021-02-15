import torch
import torch.nn as nn
from utils import get_data, encode_data, split_data
from dataloader import make_dataset


def train(BATCH_SIZE,NUM_EPOCH,NUM_WORKERS=8):

	img_paths,targets=get_data()
	encode_targets=encode_targets(targets)
	X_train,y_train,X_val,y_val,X_test,y_test=split_data(img_paths,encode_targets)

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
	optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

	model=
	
	for epoch in range(NUM_EPOCH):
		model.train()
		for X_train,y_train in train_loader:
			

