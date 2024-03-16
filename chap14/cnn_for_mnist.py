import logging
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Subset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s %(message)s')

def train(model, num_epochs, train_dl, valid_dl, loss_fn, optimizer):
    loss_hist_train = [0] * num_epochs
    accuracy_hist_train = [0] * num_epochs
    loss_hist_valid = [0] * num_epochs
    accuracy_hist_valid = [0] * num_epochs
    for epoch in range(num_epochs):
        model.train()
        for x_batch, y_batch in train_dl:
            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_hist_train[epoch] += loss.item() * y_batch.size(0)
            is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
            accuracy_hist_train[epoch] += is_correct.sum()
        loss_hist_train[epoch] /= len(train_dl.dataset)
        accuracy_hist_train[epoch] /= len(train_dl.dataset)

        model.eval()
        with torch.no_grad():
            for x_batch, y_batch in valid_dl:
                pred = model(x_batch)
                loss = loss_fn(pred, y_batch)
                loss_hist_valid[epoch] += loss.item() * y_batch.size(0)
                is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
                accuracy_hist_valid[epoch] += is_correct.sum()
        loss_hist_valid[epoch] /= len(valid_dl.dataset)
        accuracy_hist_valid[epoch] /= len(valid_dl.dataset)

        logging.info(f"Epoch {epoch + 1} accuracy: {accuracy_hist_train[epoch]:.4f} val_accuracy: {accuracy_hist_valid[epoch]:.4f}")
    return loss_hist_train, loss_hist_valid, accuracy_hist_train, accuracy_hist_valid

def main():
    logging.info("cnn_for_mnist.main()")

    image_path = r"C:\Users\sebas\Documents\datasets"
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    mnist_dataset = torchvision.datasets.MNIST(root=image_path, train=True,
                                               transform=transform, download=False)
    mnist_valid_dataset = Subset(mnist_dataset, torch.arange(10000))
    mnist_train_dataset = Subset(mnist_dataset, torch.arange(10000, len(mnist_dataset)))
    mnist_test_dataset = torchvision.datasets.MNIST(root=image_path, train=False,
                                                    transform=transform, download=False)

    batch_size = 64
    torch.manual_seed(1)
    train_dl = DataLoader(mnist_train_dataset, batch_size, shuffle=True)
    valid_dl = DataLoader(mnist_valid_dataset, batch_size, shuffle=False)

    model = torch.nn.Sequential()
    model.add_module(
        'conv1', torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
    )
    model.add_module('relu1', torch.nn.ReLU())
    model.add_module('pool1', torch.nn.MaxPool2d(kernel_size=2))
    model.add_module(
        'conv2', torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
    )
    model.add_module('relu2', torch.nn.ReLU())
    model.add_module('pool2', torch.nn.MaxPool2d(kernel_size=2))
    model.add_module('flatten', torch.nn.Flatten())
    model.add_module('fc1', torch.nn.Linear(3136, 1024))
    model.add_module('relu3', torch.nn.ReLU())
    model.add_module('dropout', torch.nn.Dropout(p=0.5))
    model.add_module('fc2', torch.nn.Linear(1024, 10))

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 20
    hist = train(model, num_epochs, train_dl, valid_dl, loss_fn, optimizer)

    x_arr = np.arange(len(hist[0])) + 1
    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(x_arr, hist[0], '-o', label='Train loss')
    ax.plot(x_arr, hist[1], '--<', label='Validation loss')
    ax.legend(fontsize=15)
    ax = fig.add_subplot(1, 2, 2)
    ax.plot(x_arr, hist[2], '-o', label='Train acc.')
    ax.plot(x_arr, hist[3], '--<', label='Validation acc.')
    ax.legend(fontsize=15)
    ax.set_xlabel('Epoch', size=15)
    ax.set_ylabel('Accuracy', size=15)
    plt.show()

    pred = model(mnist_test_dataset.data.unsqueeze(1)/255.)
    is_correct = (torch.argmax(pred, dim=1) == mnist_test_dataset.targets).float()
    logging.info(f"Test accuracy: {is_correct.mean():.4f}")

    fig = plt.figure(figsize=(12, 4))
    for i in range(12):
        ax = fig.add_subplot(2, 6, i+1)
        ax.set_xticks([]); ax.set_yticks([])
        img = mnist_test_dataset[i][0][0, :, :]
        pred = model(img.unsqueeze(0).unsqueeze(1))
        y_pred = torch.argmax(pred)
        ax.imshow(img, cmap='gray_r')
        ax.text(0.9, 0.1, y_pred.item(),
                size=15, color='blue',
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax.transAxes)
    plt.show()

if __name__ == '__main__':
    main()