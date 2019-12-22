import os
import shutil

import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn # Add on classifier
from torch import optim # Loss and optimizer


class HouseDataset(Dataset):
    """
    Custom dataset class for Houses dataset
    Index through each tiled house and its price
    """

    def __init__(self, df_x, df_y, path, transform=None):
        self.path = path
        self.transform = transform
        self.df_x = df_x
        self.df_y = df_y

    def __len__(self):
        return len(self.df_x)

    def __getitem__(self, idx):
        # given that the response variable is the house price
        # we will pass that along with the respective house
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.path,
                                self.df_x[['image']].iloc[idx][0])
        image = io.imread(img_name)
        target = self.df_y[['price']].iloc[idx][0]
        sample = {'image': image, 'target': target}

        if self.transform:
            sample = self.transform(sample)

        return sample


def train_reg_model(model, criterion, optimizer, num_epochs=10):
    """
    Training model for regression
    :param model:
    :param criterion:
    :param optimizer:
    :param num_epochs:
    :return:
    """
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_MAE = 0.0

    avg_loss = 0
    avg_MAE = 0
    avg_loss_val = 0
    avg_MAE_val = 0

    train_batches = len(dataloaders['train'])
    val_batches = len(dataloaders['val'])

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs))
        print('-' * 10)

        loss_train = 0
        loss_val = 0
        MAE_train = 0
        MAE_val = 0

        model.train(True)

        for i, data in enumerate(dataloaders['train']):
            if i % 100 == 0:
                print("\rTraining batch {}/{}".format(i, train_batches / 2), end='', flush=True)

            inputs, labels = data['image'], data['target'].type(torch.FloatTensor)

            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            loss_train += loss.item()
            MAE_train += torch.sum(torch.abs(labels.data - outputs))

            del inputs, labels, outputs
            torch.cuda.empty_cache()

        print()
        avg_loss = loss_train / len(dataloaders['train'])
        avg_MAE = MAE_train / len(dataloaders['train'])

        model.train(False)
        model.eval()

        for i, data in enumerate(dataloaders['val']):
            if i % 100 == 0:
                print("\rValidation batch {}/{}".format(i, val_batches), end='', flush=True)

            inputs, labels = data['image'], data['target']

            if use_gpu:
                inputs, labels = Variable(inputs.cuda(), volatile=True), Variable(labels.cuda(), volatile=True)
            else:
                inputs, labels = Variable(inputs, volatile=True), Variable(labels, volatile=True)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss_val += loss.item()
            MAE_val += torch.sum(torch.abs(labels.data - outputs))

            del inputs, labels, outputs, preds
            torch.cuda.empty_cache()

        avg_loss_val = loss_val / len(dataloaders['val'])
        avg_MAE_val = MAE_val / len(dataloaders['val'])

        print()
        print("Epoch {} result: ".format(epoch))
        print("Avg loss (train): {:.4f}".format(avg_loss))
        print("Avg MAE (train): {:.4f}".format(avg_MAE))
        print("Avg loss (val): {:.4f}".format(avg_loss_val))
        print("Avg MAE (val): {:.4f}".format(avg_MAE_val))
        print('-' * 10)
        print()

        if avg_MAE_val > best_MAE:
            best_acc = avg_acc_val
            best_model_wts = copy.deepcopy(vgg.state_dict())

    elapsed_time = time.time() - since
    print()
    print("Training completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    print("Best acc: {:.4f}".format(best_acc))

    model.load_state_dict(best_model_wts)
    return model