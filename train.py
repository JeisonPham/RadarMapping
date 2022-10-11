from comet_ml import Experiment
import torch
import torch.nn as nn
import torch.utils.data as torchdata
from Model import MapModel
from MapDataset import MapDataset
import os
import json
from datetime import datetime


def run_epoch(dataloader, model, loss_fn, opt, device, eval=False):
    loss = 0
    for batchii, (x, y) in enumerate(dataloader):
        if not eval:
            opt.zero_grad()

        pred = model(x.to(device))
        loss_batch = loss_fn(pred, y.to(device))

        if not eval:
            loss_batch.backward()
            opt.step()

        loss += loss_batch.detach().item()

    return loss / (batchii + 1)


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
            train_loss = run_epoch(train_dataloader, model, loss_fn, opt, device)
            experiment.log_curve("Training_Loss", train_loss, epoch)

            if epoch % 5 == 0:
                model.eval()
                with torch.no_grad():
                    test_loss = run_epoch(test_dataloader, model, loss_fn, opt, device, True)
                experiment.log_curve("Testing Loss", test_loss, epoch)
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
