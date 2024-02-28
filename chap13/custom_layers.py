# Cf. p. 426
import logging
import torch
import numpy as np
from mlxtend.plotting import plot_decision_regions
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s %(message)s')

class NoisyLinear(torch.nn.Module):
    def __init__(self, input_size, output_size, noise_stddev=0.1):
        super().__init__()
        w = torch.Tensor(input_size, output_size)
        self.w = torch.nn.Parameter(w)  # torch.nn.Parameter is a tensor that's a module parameter
        torch.nn.init.xavier_uniform(self.w)
        b = torch.Tensor(output_size).fill_(0)
        self.b = torch.nn.Parameter(b)
        self.noise_stddev = noise_stddev

    def forward(self, x, training=False):
        if training:
            noise = torch.normal(0.0, self.noise_stddev, x.shape)
            x_new = torch.add(x, noise)
        else:
            x_new = x
        return torch.add(torch.mm(x_new, self.w), self.b)

class MyNoisyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = NoisyLinear(2, 4, 0.07)
        self.a1 = torch.nn.ReLU()
        self.l2 = torch.nn.Linear(4, 4)
        self.a2 = torch.nn.ReLU()
        self.l3 = torch.nn.Linear(4, 1)
        self.a3 = torch.nn.Sigmoid()

    def forward(self, x, training=False):
        x = self.l1(x, training)
        x = self.a1(x)
        x = self.l2(x)
        x = self.a2(x)
        x = self.l3(x)
        x = self.a3(x)
        return x

    def predict(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        pred = self.forward(x)[:, 0]
        return (pred >= 0.5).float()


def main():
    logging.info("custom_layers.main()")

    torch.manual_seed(1)
    model = MyNoisyModule()

    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.015)
    torch.manual_seed(1)

    x = np.random.uniform(low=-1, high=1, size=(200, 2))
    y = np.ones(len(x))
    y[x[:, 0] * x[:, 1] < 0] = 0
    n_train = 100
    x_train = torch.tensor(x[: n_train, :], dtype=torch.float32)
    y_train = torch.tensor(y[: n_train], dtype=torch.float32)
    x_valid = torch.tensor(x[n_train:, :], dtype=torch.float32)
    y_valid = torch.tensor(y[n_train:], dtype=torch.float32)

    train_ds = TensorDataset(x_train, y_train)
    batch_size = 2
    torch.manual_seed(1)
    train_dl = DataLoader(train_ds, batch_size, shuffle=True)

    num_epochs = 200
    loss_hist_train = [0] * num_epochs
    accuracy_hist_train = [0] * num_epochs
    loss_hist_valid = [0] * num_epochs
    accuracy_hist_valid = [0] * num_epochs
    for epoch in range(num_epochs):
        for x_batch, y_batch in train_dl:
            pred = model(x_batch, True)[:, 0]
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

    fig = plt.figure(figsize=(16, 4))
    ax = fig.add_subplot(1, 3, 1)
    plt.plot(loss_hist_train, lw=4)
    plt.plot(loss_hist_valid, lw=4)
    plt.legend(['Train loss', 'Validation loss'], fontsize=15)
    ax.set_xlabel('Epochs', size=15)
    ax = fig.add_subplot(1, 3, 2)
    plt.plot(accuracy_hist_train, lw=4)
    plt.plot(accuracy_hist_valid, lw=4)
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