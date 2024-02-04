# Cf. p. 344
import logging
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from neuralnet import NeuralNetMLP, int_to_onehot

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s %(message)s')

def minibatch_generator(X, y, minibatch_size):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    for start_idx in range(0, indices.shape[0] - minibatch_size + 1, minibatch_size):
        batch_idx = indices[start_idx: start_idx + minibatch_size]
        yield X[batch_idx], y[batch_idx]

def mse_loss(targets, probas, num_labels=10):
    onehot_targets = int_to_onehot(targets, num_labels=num_labels)
    return np.mean((onehot_targets - probas)**2)

def accuracy(targets, predicted_labels):
    return np.mean(predicted_labels == targets)

def compute_mse_and_acc(nnet, X, y, num_labels=10, minibatch_size=100):
    mse, correct_pred, num_examples = 0., 0, 0
    minibatch_gen = minibatch_generator(X, y, minibatch_size)
    for i, (features, targets) in enumerate(minibatch_gen):
        _, probas = nnet.forward(features)
        predicted_labels = np.argmax(probas, axis=1)
        onehot_targets = int_to_onehot(targets, num_labels=num_labels)
        loss = np.mean((onehot_targets - probas)**2)
        correct_pred += (predicted_labels == targets).sum()
        num_examples += targets.shape[0]
        mse += loss
    mse = mse/i
    acc = correct_pred/num_examples
    return mse, acc

def train(model, X_train, y_train, X_valid, y_valid, num_epochs, learning_rate=0.1, minibatch_size=100):  # Cf. p. 354
    epoch_loss = []
    epoch_train_acc = []
    epoch_valid_acc = []

    for e in range(num_epochs):
        # Iterate over minibatches
        minibatch_gen = minibatch_generator(X_train, y_train, minibatch_size=minibatch_size)
        for X_train_mini, y_train_mini in minibatch_gen:
            # Compute outputs
            a_h, a_out = model.forward(X_train_mini)
            # Compute gradients
            d_loss__d_w_out, d_loss__d_b_out, d_loss__d_w_h, d_loss__d_b_h = \
                model.backward(X_train_mini, a_h, a_out, y_train_mini)
            # Update weights
            model.weight_h -= learning_rate * d_loss__d_w_h
            model.bias_h -= learning_rate * d_loss__d_b_h
            model.weight_out -= learning_rate * d_loss__d_w_out
            model.bias_out -= learning_rate * d_loss__d_b_out

        # Epoch logging
        train_mse, train_acc = compute_mse_and_acc(model, X_train, y_train)
        valid_mse, valid_acc = compute_mse_and_acc(model, X_valid, y_valid)
        train_acc, valid_acc = train_acc*100, valid_acc*100
        epoch_train_acc.append(train_acc)
        epoch_valid_acc.append(valid_acc)
        epoch_loss.append(train_mse)
        print(f'Epoch: {e+1:03d}/{num_epochs:03d} '
              f'| Train MSE: {train_mse:.2f} '
              f'| Train Acc: {train_acc:.2f}% '
              f'| Valid Acc: {valid_acc:.2f}%')
    return epoch_loss, epoch_train_acc, epoch_valid_acc


def main():
    logging.info("classification_mnist.main()")

    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    X = X.values
    y = y.astype(int).values
    logging.info(f"X.shape = {X.shape}; y.shape = {y.shape}")
    X = ((X / 255.) - 0.5) * 2  # [0, 255] -> [-1, 1]

    fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)
    ax = ax.flatten()
    for i in range(10):
        img = X[y==i][0].reshape(28, 28)
        ax[i].imshow(img, cmap='Greys')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True)
    ax = ax.flatten()
    for i in range(25):
        img = X[y==7][i].reshape(28, 28)
        ax[i].imshow(img, cmap='Greys')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=10000, random_state=123, stratify=y)
    X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=5000, random_state=123, stratify=y_temp)

    num_epochs = 50
    minibatch_size = 100

    model = NeuralNetMLP(num_features=28*28, num_hidden=50, num_classes=10)
    _, probas = model.forward(X_valid)
    """mse = mse_loss(y_valid, probas)
    predicted_labels = np.argmax(probas, axis=1)
    acc = accuracy(y_valid, predicted_labels)
    """
    mse, acc = compute_mse_and_acc(model, X_valid, y_valid)
    logging.info(f"Initial validation MSE: {mse:.3f}")
    logging.info(f"Initial validation accuracy: {acc*100:.1f}%")

    np.random.seed(123)
    epoch_loss, epoch_train_acc, epoch_valid_acc = train(model, X_train, y_train, X_valid, y_valid,
                                                         num_epochs=50, learning_rate=0.1)
    plt.plot(range(len(epoch_loss)), epoch_loss)
    plt.ylabel('Mean squared error')
    plt.xlabel('Epoch')
    plt.show()

    plt.plot(range(len(epoch_train_acc)), epoch_train_acc, label='Training')
    plt.plot(range(len(epoch_valid_acc)), epoch_valid_acc, label='Validation')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    plt.show()

    test_mse, test_acc = compute_mse_and_acc(model, X_test, y_test)
    logging.info(f"Test accuracy: {test_acc*100:.2f}%")

    # Misclassified examples
    X_test_subset = X_test[:1000, :]
    y_test_subset = y_test[:1000]
    _, probas = model.forward(X_test_subset)
    test_pred = np.argmax(probas, axis=1)
    misclassified_images = \
        X_test_subset[y_test_subset != test_pred][:25]
    misclassified_labels = test_pred[y_test_subset != test_pred][:25]
    correct_labels = y_test_subset[y_test_subset != test_pred][:25]

    fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True, figsize=(8, 8))
    ax = ax.flatten()
    for i in range(25):
        img = misclassified_images[i].reshape(28, 28)
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')
        ax[i].set_title(f'{i+1} '
                        f'True: {correct_labels[i]}\n'
                        f' Predicted: {misclassified_labels[i]}')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()