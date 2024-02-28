# Cf. p. 424
import logging
import torch
from mlxtend.plotting import plot_decision_regions
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s %(message)s')

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        l1 = torch.nn.Linear(2, 8)
        a1 = torch.nn.ReLU()
        l2 = torch.nn.Linear(8, 8)
        a2 = torch.nn.ReLU()
        l3 = torch.nn.Linear(8, 1)
        a3 = torch.nn.Sigmoid()
        l = [l1, a1, l2, a2, l3, a3]
        self.module_list = torch.nn.ModuleList(l)

    def forward(self, x):
        for f in self.module_list:
            x = f(x)
        return x

    def predict(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        pred = self.forward(x)[:, 0]
        return (pred >= 0.5).float()

def train(model, num_epochs, train_dl, x_valid, y_valid, loss_fn, optimizer, n_train, batch_size):
    loss_hist_train = [0] * num_epochs
    accuracy_hist_train = [0] * num_epochs
    loss_hist_valid = [0] * num_epochs
    accuracy_hist_valid = [0] * num_epochs
    for epoch in range(num_epochs):
        for x_batch, y_batch in train_dl:
            pred = model(x_batch)[:, 0]
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_hist_train[epoch] += loss.item()
            is_correct = ((pred >= 0.5).float() == y_batch).float()
            accuracy_hist_train[epoch] += is_correct.mean()
        loss_hist_train[epoch] /= n_train/batch_size
        accuracy_hist_train[epoch] /= n_train/batch_size
        pred = model(x_valid)[:, 0]
        loss = loss_fn(pred, y_valid)
        loss_hist_valid[epoch] = loss.item()
        is_correct = ((pred >= 0.5).float() == y_valid).float()
        accuracy_hist_valid[epoch] += is_correct.mean()
    return loss_hist_train, loss_hist_valid, accuracy_hist_train, accuracy_hist_valid



def main():
    logging.info("model_with_nn.Module.main()")

    torch.manual_seed(1)
    np.random.seed(1)

    x = np.random.uniform(low=-1, high=1, size=(200, 2))
    y = np.ones(len(x))
    y[x[:, 0] * x[:, 1] < 0] = 0
    n_train = 100
    x_train = torch.tensor(x[: n_train, :], dtype=torch.float32)
    y_train = torch.tensor(y[: n_train], dtype=torch.float32)
    x_valid = torch.tensor(x[n_train:, :], dtype=torch.float32)
    y_valid = torch.tensor(y[n_train:], dtype=torch.float32)

    model = MyModule()
    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.015)

    train_ds = TensorDataset(x_train, y_train)
    batch_size = 2
    torch.manual_seed(1)
    train_dl = DataLoader(train_ds, batch_size, shuffle=True)

    torch.manual_seed(1)
    num_epochs = 200

    history = train(model, num_epochs, train_dl, x_valid, y_valid, loss_fn, optimizer, n_train, batch_size)

    fig = plt.figure(figsize=(16, 4))
    ax = fig.add_subplot(1, 3, 1)
    plt.plot(history[0], lw=4)
    plt.plot(history[1], lw=4)
    plt.legend(['Train loss', 'Validation loss'], fontsize=15)
    ax.set_xlabel('Epochs', size=15)
    ax = fig.add_subplot(1, 3, 2)
    plt.plot(history[2], lw=4)
    plt.plot(history[3], lw=4)
    plt.legend(['Train acc.', 'Validation acc.'], fontsize=15)
    ax.set_xlabel('Epochs', size=15)
    ax = fig.add_subplot(1, 3, 3)
    plot_decision_regions(X=x_valid.numpy(), y=y_valid.numpy().astype(np.integer), clf=model)
    ax.set_xlabel(r'$x_1$', size=15)
    ax.xaxis.set_label_coords(1, -0.025)
    ax.set_ylabel(r'$x_2$', size=15)
    ax.yaxis.set_label_coords(-0.025, 1)
    plt.show()


if __name__ == '__main__':
    main()