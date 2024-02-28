# Cf. p. 387
import logging
import torchvision
from itertools import islice
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s %(message)s')

def main():
    logging.info("celeba_and_mnist.main()")
    celeba_dataset = torchvision.datasets.CelebA(
        root=r'C:\Users\sebas\Documents\datasets', split='train', target_type='attr', download=False
    )

    # Display celeb images with their attribute 'smiling', i.e. attributes[31]
    fig = plt.figure(figsize=(12, 8))
    for i, (image, attributes) in islice(enumerate(celeba_dataset), 18):
        ax = fig.add_subplot(3, 6, i+1)
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(image)
        ax.set_title(f'{attributes[31]}', size=15)
    plt.show()

    mnist_dataset = torchvision.datasets.MNIST(r'C:\Users\sebas\Documents\datasets', 'train',
                                               download=False)
    fig = plt.figure(figsize=(15, 6))
    for i, (image, label) in islice(enumerate(mnist_dataset), 10):
        ax = fig.add_subplot(2, 5, i+1)
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(image, cmap='gray_r')
        ax.set_title(f'{label}', size=15)
    plt.show()

if __name__ == '__main__':
    main()