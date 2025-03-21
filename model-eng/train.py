import torch

def train(dataloader, model, loss_fn, optimizer, cumulative_metrics, device, task_type):
    model.train()
    cumulative_metrics.reset()
    for _, (img, binary, _, hb_level, _) in enumerate(dataloader):
        img, binary = img.to(device), binary.to(device).unsqueeze(1)
        optimizer.zero_grad()
        pred = model(img)
        loss = loss_fn(pred, binary)
        loss.backward()
        optimizer.step()
        cumulative_metrics.update(pred, binary)
    return cumulative_metrics.compute()

def eval(dataloader, model, loss_fn, cumulative_metrics, device, task_type):
    model.eval()
    cumulative_metrics.reset()
    with torch.no_grad():
        for _, (img, binary, _, hb_level, _) in enumerate(dataloader):
            img, binary = img.to(device), binary.to(device).unsqueeze(1)
            pred = model(img)
            loss = loss_fn(pred, binary)
            cumulative_metrics.update(pred, binary)
    return cumulative_metrics.compute()
