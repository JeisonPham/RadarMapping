from comet_ml import Experiment
import torch
import torch.nn as nn
import torch.utils.data as torchdata
from Model import MapModel
from MapDataset import MapDataset
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import util
import time
import numpy as np
from focalloss import FocalLoss

def to_numpy(x):
    return x.detach().cpu().numpy()
    


def train_epoch(epoch, dataloader, model, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for batchii, (x, y) in enumerate(dataloader):
        optimizer.zero_grad()
        x = x.float()
        y = y.float()
        output = model(x.to(device))
        y = y.to(device)
        loss = 0.25 * loss_fn(output[:, :2, :, :], y.to(device)[:, :2, :, :]) + 0.75 * loss_fn(output[:, 2, :, :][y[:, 0] == 1], y[:, 2, :, :][y[:, 0] == 1])
        loss.backward()
        optimizer.step()
        

        total_loss += loss.detach().cpu().item()

        if batchii % 1000 == 0:
            print(f"Train epoch: [{epoch}/{batchii}] \t Loss: {loss.item():.6f}")
            
    return total_loss / (batchii + 1)


def test_epoch(epoch, dataloader, model, loss_fn, device):
    model.eval()
    test_loss = 0
    test_acc = 0
    loss_index = []
    acc_index = []
    data_values = []
    
    with torch.no_grad():
        for batchii, (x, y) in enumerate(dataloader):
            x = x.float()
            y = y.float()
            output = model(x.to(device))
            y = y.to(device)
            loss = 0.25 * loss_fn(output[:, :2, :, :], y[:, :2, :, :]) + 0.75 * loss_fn(output[:, 2, :, :][y[:, 0] == 1], y[:, 2, :, :][y[:, 0] == 1])
            test_loss += loss.detach().cpu().item()
            
            loss_index.append(loss.detach().cpu().item() / x.shape[0])
            data_values.append((loss.detach().cpu().item() / x.shape[0], x.detach().cpu().numpy(), y.detach().cpu().numpy(), output.detach().cpu().numpy()))
            
            output = np.where(to_numpy(output) >= 0.5, 1., 0.)
            acc = (output == to_numpy(y)).astype(int).mean()
            acc_index.append(acc)
            test_acc += acc
            
    indexes = np.argsort(loss_index)
    indexes1 = np.argsort(acc_index)
    
    data = dict(
      first_loss=data_values[0],
      last_loss = data_values[-1],
      first_acc = data_values[0],
      last_acc = data_values[-1]
    )

    return data, test_loss / (batchii + 1), test_acc / (batchii + 1)


def train(params, device):
    

    device = torch.device(device)
    PD = MapDataset(**params['train']['dataset_params'])

    N = len(PD)
    print(N)
    train_len = int(N * 0.8)
    test_len = N - train_len

    train_set, test_set = torchdata.random_split(PD, [train_len, test_len])
    train_dataloader = torchdata.DataLoader(train_set, batch_size=params['train']['batch_size'], shuffle=True,
                                            num_workers=params['train']['num_workers'])
    test_dataloader = torchdata.DataLoader(test_set, batch_size=params['train']['batch_size'], shuffle=False,
                                           num_workers=params['train']['num_workers'])

    model = MapModel(3 * PD.time_window, 3, PD.nx, PD.nx).to(device)
    if "load_model_location" in params['train']:
        model.load_state_dict(torch.load(params['train']['load_model_location']))
    model.train()

    lr = params['train']['hyper_params']['lr']
    wd = params['train']['hyper_params']['weight_decay']
    momentum = params['train']['hyper_params'].get('momentum', 0)
    epochs = params['train']['hyper_params']['epoch']
    model_location = params['train']['model_location']

    loss_fn = nn.BCELoss()
    opt = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=momentum)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min")

    counter = 0
    
    experiment = Experiment(**params['comet'], disabled=params['train']['disable'])
    experiment.log_parameters(params['train']['hyper_params'])
    
    fig = plt.figure(figsize=(18, 6))
    with experiment.train():
        for epoch in range(int(epochs)):
            print(f"Starting Epoch {epoch}")
            train_loss = train_epoch(epoch, train_dataloader, model, opt, loss_fn, device)
            experiment.log_metric("train_Loss", train_loss, step=epoch)
            
            scheduler.step(train_loss)
            experiment.log_metric("train/lr", opt.param_groups[0]['lr'], step=epoch)

            if epoch % 5 == 0:
                data, test_loss, test_acc = test_epoch(epoch, test_dataloader, model, loss_fn, device)
                experiment.log_metric("Test_Loss", test_loss, step=epoch)
                experiment.log_metric("Test_Acc", test_acc, step=epoch)
                
                for key, (score, x, gt, pred) in data.items():
                  for i, (xi, gti, predi) in enumerate(zip(x, gt, pred)):
                      util.render_maps(xi, gti, predi)
                      plt.savefig("temp.png")
                      experiment.log_image("temp.png", name=f"{key}\tIndex: {i}", step=epoch)
                      plt.clf()
                      time.sleep(2)

                model.train()
                scheduler.step(test_loss)
                experiment.log_metric("train/lr", opt.param_groups[0]['lr'], step=epoch)

            if epoch % 5 == 0:
                model.eval()
                mname = os.path.join(model_location, f"{epoch} {datetime.now().strftime('%Y_%m_%d %H-%M-%S')}.pt")
                print("saving", mname)
                torch.save(model.state_dict(), mname)
                model.train()
                
def test(params, device):
    device = torch.device(device)
    PD = MapDataset(**params['train']['dataset_params'])
    
    N = len(PD)
    train_len = int(N * 0.8)
    test_len = N - train_len

    train_set, test_set = torchdata.random_split(PD, [train_len, test_len])
    train_dataloader = torchdata.DataLoader(train_set, batch_size=params['train']['batch_size'], shuffle=True,
                                            num_workers=params['train']['num_workers'])
    test_dataloader = torchdata.DataLoader(test_set, batch_size=params['train']['batch_size'], shuffle=False,
                                           num_workers=params['train']['num_workers'])

    model = MapModel(3, 1, PD.nx, PD.nx).to(device)
    model.load_state_dict(torch.load(params['test']['model_location']))
    model.eval()
    
    if not os.path.exists(params['test']['save_location']):
      os.mkdir(params['test']['save_location'])
    
    counter = 0
    fig = plt.figure(figsize=(12, 6))
    with torch.no_grad():
        for batchii, (x, y) in enumerate(test_dataloader):
            x = x.float()
            y = y.float()
            
            output = model(x.to(device))
            
            x = x.detach().cpu().numpy()
            y = y.detach().cpu().numpy()
            output = output.detach().cpu().numpy()
            
            for xi, gti, predi in zip(x, y, output):
                util.render_maps(xi, gti, predi)
                plt.savefig(os.path.join(params['test']['save_location'], f"RadarMap_{counter:05d}.png"))
                counter += 1
                plt.clf()

                


if __name__ == "__main__":
    import toml

    with open("config.toml", "r") as file:
        params = toml.load(file)
    print(params)
    
    if not os.path.exists(params['train']['model_location']):
      os.mkdir(params['train']['model_location'])

    device = torch.device('cuda' if params['train']['device'] != 0 and torch.cuda.is_available() else 'cpu')
    train(params, device)
