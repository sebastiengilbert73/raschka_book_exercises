# Cf. p. 394
import logging
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s %(message)s')

def main():
    logging.info("model_training_with_pytorch.main()")

    torch.manual_seed(1)

    X_train = np.arange(10, dtype='float32').reshape((10, 1))
    y_train = np.array([1.0, 1.3, 3.1, 2.0, 5.0, 6.3, 6.6, 7.4, 8.0, 9.0], dtype='float32')

    X_train_norm = (X_train - np.mean(X_train)) / np.std(X_train)
    X_train_norm = torch.from_numpy(X_train_norm)
    y_train = torch.from_numpy(y_train).float()
    train_ds = TensorDataset(X_train_norm, y_train)
    batch_size = 1
    train_dl = DataLoader(train_ds, batch_size, shuffle=True)

    learning_rate = 0.001
    num_epochs = 200
    log_epochs = 10

    loss_fn = nn.MSELoss(reduction='mean')
    input_size = 1
    output_size = 1
    model = nn.Linear(input_size, output_size)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for x_batch, y_batch in train_dl:
            # 1. Generate predictions
            pred = model(x_batch)[:, 0]
            # 2. Calculate loss
            loss = loss_fn(pred, y_batch)
            # 3. Compute gradients
            loss.backward()
            # 4. Update parameters using gradients
            optimizer.step()
            # 5. Reset the gradients to zero
            optimizer.zero_grad()
        if epoch % log_epochs == 0:
            logging.info(f'Epoch {epoch}  Loss: {loss.item():.4f}')

    logging.info(f'Final parameters: {model.weight.item()}; {model.bias.item()}')


if __name__ == '__main__':
    main()