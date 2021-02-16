from tqdm import tqdm
from loss import ctc_loss



def train_fn(model, data_loader, optimizer,device):
    model.train()
    fin_loss = 0
    tk0 = tqdm(data_loader, total=len(data_loader))
    for X,y in tk0:
        X=X.to(device)
        y=y.to(device)
        #print(y.shape)
        optimizer.zero_grad()
        y_hat = model(X)
        loss = ctc_loss(y_hat, y)
        loss.backward()
        optimizer.step()
        fin_loss += loss.item()
    return fin_loss / len(data_loader)


def eval_fn(model, data_loader,device):
    model.eval()
    fin_loss = 0
    tk0 = tqdm(data_loader, total=len(data_loader))
    for X,y in tk0:
        X=X.to(device)
        y=y.to(device)
        y_hat=model(X)
        loss=ctc_loss(y_hat, y)
        fin_loss += loss.item()
    return fin_loss / len(data_loader)