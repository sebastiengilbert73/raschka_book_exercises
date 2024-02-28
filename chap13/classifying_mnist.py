# Cf. p. 437
import logging
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s %(message)s')

def main():
    logging.info("classifying_mnist.main()")

    image_path = r"C:\Users\sebas\Documents\datasets"
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_train_dataset = torchvision.datasets.MNIST(root=image_path, train=True, transform=transform,
                                                     download=False)
    mnist_test_dataset = torchvision.datasets.MNIST(root=image_path, train=False, transform=transform,
                                                     download=False)
    batch_size = 64
    torch.manual_seed(1)
    train_dl = DataLoader(mnist_train_dataset, batch_size, shuffle=True)

    # Create the model
    hidden_units = [32, 16]
    image_size = mnist_train_dataset[0][0].shape  # (1, 28, 28)
    input_size = image_size[0] * image_size[1] * image_size[2]
    all_layers = [torch.nn.Flatten()]
    for hidden_unit in hidden_units:
        layer = torch.nn.Linear(input_size, hidden_unit)
        all_layers.append(layer)
        all_layers.append(torch.nn.ReLU())
        input_size = hidden_unit
    all_layers.append(torch.nn.Linear(hidden_units[-1], 10))
    model = torch.nn.Sequential(*all_layers)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    torch.manual_seed(1)
    num_epochs = 20
    for epoch in range(num_epochs):
        accuracy_hist_train = 0
        for x_batch, y_batch in train_dl:
            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
            accuracy_hist_train += is_correct.sum()
        accuracy_hist_train /= len(train_dl.dataset)
        logging.info(f"Epoch {epoch}: Accuracy {accuracy_hist_train:.4f}")

    pred = model(mnist_test_dataset.data / 255.)  # mnist_test_dataset.data is a tensor (10000, 28, 28) with values in [0, 255]
    is_correct = (torch.argmax(pred, dim=1) == mnist_test_dataset.targets).float()
    logging.info(f"Test accuracy: {is_correct.mean():.4f}")

if __name__ == '__main__':
    main()