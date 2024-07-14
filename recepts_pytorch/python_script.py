#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import tqdm
import torch
import torch.nn.functional as F
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score as r2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch import optim
from torch.optim import Optimizer
np.random.seed(42)
torch.manual_seed(42)
df = pd.read_csv('recepts.csv')
df.head()
df.isna().sum()
print(df.dtypes)
print(df.info())
print(df.shape)
Y = df['calories']
X = df.iloc[:, 0:44]
y = df.iloc[:, 44]
X.head()
print(y[:5])
print(X.describe())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
scaler = MinMaxScaler()
regressor = RandomForestRegressor(n_estimators=100,
    criterion='poisson', max_depth=14, max_features=9, n_jobs = 8, random_state=40)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
r2(y_test, y_pred)
print(r2(y_test, y_pred))
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_tensor = torch.Tensor(np.array(X_train_scaled))
X_test_tensor = torch.Tensor(np.array(X_test_scaled))
y_train_tensor = torch.Tensor(np.array(y_train))
y_test_tensor = torch.Tensor(np.array(y_test))
n_data, n_features = X_train_tensor.shape
print(n_data)
print(n_features)
def loss(mean, target):
    """function loss"""
    return mean(F.l1_loss(input, target, reduction="none") / target) * 100
loss_func = F.mse_loss
metrics_func_1 = [loss_func, loss]
metrics_name = ["MSE", "LOSS"]
def print_metrics(models, train_data, test_data, models_name):
    """function print_metrics"""
    results = np.ones(2 * len(models), len(metrics_func_1))
    models_name = []
    for model in models_name:
        models_name.extend([model + "Train", model + "Test"])
    for row, sample in enumerate([train_data, test_data]):
        results[row + sample * 2] = evaluate(models, metrics_func_1, sample[0], sample[1])
        results = pd.DataFrame(results, columns=metrics_name, index=models_name)
        return results
train_data_1 = (X_train_tensor, y_train_tensor)
test_data_1 = (X_test_tensor, y_test_tensor)
model_lr_sklearn = LinearRegression()
model_lr_sklearn.fit(X_train_scaled, y_train)
model_1 = [model_lr_sklearn.predict]
metrics_name = ["MSE", "LOSS"]
models_name_1 = ["LOSS"]
model_lr = nn.Sequential(
    nn.Linear(in_features=n_features,
    out_features=1))
print(model_lr)
metrics_name_1 = torch.nn.MSELoss()
Optimizer = optim.SGD(params=model_lr.parameters(), lr=0.001)
BATCH_SIZE_LR = 16
EPOCHS_LR = 1000
for epoch in tqdm.trange(EPOCHS_LR):
    for i in range((n_data - 1) // BATCH_SIZE_LR + 1):
        start_i = i * BATCH_SIZE_LR
        end_i = start_i + BATCH_SIZE_LR
        Xb = X_train_tensor[start_i: end_i]
        yb = y_train_tensor[start_i: end_i]
        pred = yb
        loss_1 = loss_func(pred, yb + 1)
print(X_test)
print(y_test)
X = torch.randn(BATCH_SIZE_LR, 44)
y = torch.Tensor([[1.0], [0.0], [0.0], [1.0], [1.0],
    [1.0], [0.0], [0.0], [1.0], [1.0],
    [1.0], [0.0], [0.0], [1.0], [1.0],
    [1.0]])
for epoch in range(1000):
    y_pred = model_lr(X)
    loss = metrics_name_1(y_pred, y)
    print('epoch: ', epoch,' loss: ', loss.item())
    Optimizer.zero_grad()
    loss.backward()
    Optimizer.step()
def train(model_lr, loss_fn, optimizer):
    """function train"""
    size = len(X_train_tensor)
    model_lr.train()
    for batch_size_lr in range(1000):
        y_pred_1 = model_lr(X)
        loss = loss_fn(y_pred_1, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if BATCH_SIZE_LR % 16 == 0:
            loss_2 = loss.item(), BATCH_SIZE_LR * len(X)
            print(f'loss: {loss_2}')
def test(model_lr, loss_fn):
    """function test"""
    size = len(X_train_tensor)
    num_batches = size
    model_lr.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_size_lr in range(1000):
            y_pred_2 = model_lr(X)
            test_loss += loss_fn(y_pred_2, y).item()
            test_loss /= num_batches
            print(f'Test loss: {test_loss}')
loss_fn = nn.CrossEntropyLoss()
for epoch in range(EPOCHS_LR):
    print(f'Epoch: {epoch + 1}')
    train(model_lr, loss_fn, Optimizer)
    test(model_lr, loss_fn)
with torch.no_grad():
    print(model_lr(X_test_tensor[-1:]))
torch.save(model_lr, "lr.pth")
model_lr_1 = nn.Sequential(
    nn.Linear(in_features=n_features, out_features=16),
    nn.ReLU(),
    nn.Linear(in_features=16, out_features=32),
    nn.ReLU(),
    nn.Linear(in_features=32, out_features=32),
    nn.ReLU(),
    nn.Linear(in_features=32, out_features=32),
    nn.ReLU(),
    nn.Linear(in_features=32, out_features=32),
    nn.ReLU(),
    nn.Linear(in_features=32, out_features=1))
print(model_lr_1)
metrics_name_2 = torch.nn.MSELoss()
Optimizer = optim.SGD(params=model_lr_1.parameters(), lr=0.0001)
BATCH_SIZE_LR_1 = 16
EPOCHS_LR_1 = 1000
for epoch in tqdm.trange(EPOCHS_LR_1):
    for i in range((n_data - 1) // BATCH_SIZE_LR_1 + 1):
        start_i = i * BATCH_SIZE_LR_1
        end_i = start_i + BATCH_SIZE_LR_1
        Xb = X_train_tensor[start_i: end_i]
        yb = y_train_tensor[start_i: end_i]
        pred_2 = yb
        loss = loss_func(pred_2, yb + 1)
print(X_test)
print(y_test)
X = torch.randn(BATCH_SIZE_LR_1, 44)
y = torch.Tensor([[1.0], [0.0], [0.0], [1.0], [1.0],
    [1.0], [0.0], [0.0], [1.0], [1.0],
    [1.0], [0.0], [0.0], [1.0], [1.0],
    [1.0]])
for epoch in range(1000):
    y_pred_3 = model_lr_1(X)
    loss = metrics_name_2(y_pred_3, y)
    print('epoch: ', epoch,' loss: ', loss.item())
    Optimizer.zero_grad()
    loss.backward()
    Optimizer.step()
def train_1(model_lr_1, loss_fn, optimizer):
    """function_1 train_1"""
    size = len(X_train_tensor)
    model_lr_1.train()
    for batch_size_lr_1 in range(1000):
        y_pred_4 = model_lr_1(X)
        loss = loss_fn(y_pred_4, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if BATCH_SIZE_LR_1 % 16 == 0:
            loss_3 = loss.item(), BATCH_SIZE_LR * len(X)
            print(f'loss: {loss_3}')
def test_1(model_lr_1, loss_fn):
    """function_1 test_1"""
    size = len(X_train_tensor)
    num_batches = size
    model_lr_1.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_size_lr_1 in range(1000):
            y_pred_5 = model_lr_1(X)
            test_loss += loss_fn(y_pred_5, y).item()
            test_loss /= num_batches
            print(f'Test loss: {test_loss}')
with torch.no_grad():
    print(model_lr_1(X_test_tensor[-1:]))
torch.save(model_lr_1, "lr_1.pth")
