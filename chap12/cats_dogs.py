import logging
import pathlib
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s %(message)s')

class ImageDataset(Dataset):
    def __init__(self, file_list, labels, transform=None):
        self.file_list = file_list
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.file_list[index])
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[index]
        return img, label

    def __len__(self):
        return len(self.labels)

def main():
    logging.info("cats_dogs.main()")

    imgdir_path = pathlib.Path(r"C:\Users\sebas\Documents\projects\machine-learning-book\ch12\cat_dog_images")
    file_list = sorted([str(path) for path in imgdir_path.glob('*.jpg')])
    logging.info(f"file_list = {file_list}")
    fig = plt.figure(figsize=(10, 5))
    for i, file in enumerate(file_list):
        img = Image.open(file)
        logging.info(f"Image shape: {np.array(img).shape}")
        ax = fig.add_subplot(2, 3, i+1)
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(img)
        ax.set_title(os.path.basename(file), size=15)
    plt.tight_layout()
    plt.show()

    labels = [1 if 'dog' in os.path.basename(file) else 0 for file in file_list]
    logging.info(f"labels = {labels}")

    img_height, img_width = 80, 120
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_height, img_width))
    ])

    image_dataset = ImageDataset(file_list, labels, transform)
    fig = plt.figure(figsize=(10, 6))
    for i, example in enumerate(image_dataset):
        ax = fig.add_subplot(2, 3, i+1)
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(example[0].numpy().transpose((1, 2, 0)))
        ax.set_title(f'{example[1]}', size=15)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()