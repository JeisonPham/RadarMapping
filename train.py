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


def train_epoch(epoch, dataloader, model, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for batchii, (x, y) in enumerate(dataloader):
        optimizer.zero_grad()
        output = model(x.to(device))
        loss = loss_fn(y.to(device), output)
        loss.backward()
        optimizer.step()

        total_loss += loss.detach().item()

        if batchii % 50 == 0:
            print(f"Train epoch: [{epoch}/{batchii}] \t Loss: {loss.item():.6f}")
    return total_loss / (batchii + 1)


def test_epoch(epoch, dataloader, model, loss_fn, device):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batchii, (x, y) in enumerate(dataloader):
            output = model(x.to(device))
            loss = loss_fn(output, y.to(device))
            test_loss += loss.detach().item()

    return x, y, output, test_loss / (batchii + 1)


def train(params, device):
    experiment = Experiment(**params['comet'], disabled=params['train']['disable'])
    experiment.log_parameters(params['train']['hyper_params'])

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

    model = MapModel(3, 1, 256, 256)
    model.train()

    lr = params['train']['hyper_params']['lr']
    wd = params['train']['hyper_params']['weight_decay']
    epochs = params['train']['hyper_params']['epoch']
    model_location = params['train']['model_location']

    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min")

    counter = 0
    with experiment.train():
        for epoch in range(int(epochs)):
            print(f"Starting Epoch {epoch}")
            train_loss = train_epoch(epoch, train_dataloader, model, opt, loss_fn, device)
            experiment.log_curve("Training_Loss", train_loss, epoch)

            if epoch % 5 == 0:
                x, gt, pred, test_loss = test_epoch(epoch, test_dataloader, model, loss_fn, device)
                experiment.log_curve("Testing Loss", test_loss, epoch)

                for i, (xi, gti, predi) in enumerate(zip(x, gt, pred)):
                    util.render_maps(xi, gti, predi)
                    experiment.log_figure(figure_name=f"Maps for {epoch}_{i}", figure=plt)
                    plt.clf()

                model.train()
                scheduler.step(test_loss)
                experiment.log_metric("train/lr", opt.param_groups[0]['lr'], step=epoch)

            if epoch % 10 == 0:
                model.eval()
                mname = os.path.join(model_location, f"{counter} {datetime.now().strftime('%Y_%m_%d %H-%M-%S')}.pt")
                print("saving", mname)
                torch.save(model.state_dict(), mname)
                model.train()


if __name__ == "__main__":
    import toml

    with open("config.toml", "r") as file:
        params = toml.load(file)
    print(params)

    device = torch.device('cuda' if params['train']['device'] != 0 and torch.cuda.is_available() else 'cpu')
    train(params, device)
