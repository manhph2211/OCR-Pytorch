from tqdm import tqdm

def train_fn(model, data_loader, optimizer,device,loss_fn):
    model.train()
    fin_loss = 0
    tk0 = tqdm(data_loader, total=len(data_loader))
    for X,y in tk0:
        X=X.to(device)
        y=y.to(device)
        print(y.shape)
        optimizer.zero_grad()
        y_hat = model(X)
        loss = loss_fn(y_hat, y)
        loss.backward()
        optimizer.step()
        fin_loss += loss.item()
    return fin_loss / len(data_loader)


def eval_fn(model, data_loader,device,loss_fn):
    model.eval()
    fin_loss = 0
    fin_preds = []
    tk0 = tqdm(data_loader, total=len(data_loader))
    for X,y in tk0:
        X=X.to(device)
        y=y.to(device)
        y_hat=model(X)
        loss=loss_fn(y_hat, y)
        fin_loss += loss.item()
        fin_preds.append(y_hat)
    return fin_preds, fin_loss / len(data_loader)