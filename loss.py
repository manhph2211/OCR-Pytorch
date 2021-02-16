import torch
from torch.nn import functional as F
import torch.nn as nn


def ctc_loss(x,targets): # x<->y_hat
	bs=x.shape[0]
	log_probs = F.log_softmax(x, 2) # 8x13x19
	input_lengths = torch.full(
	    size=(bs,), fill_value=log_probs.size(0), dtype=torch.int32
	    )
	target_lengths = torch.full(
	    size=(bs,), fill_value=targets.size(1), dtype=torch.int32
	    )
	loss = nn.CTCLoss(blank=13)(
	    log_probs, targets, input_lengths, target_lengths
	    )
	return loss