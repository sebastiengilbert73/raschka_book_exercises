# Cf. p. 444
import logging
from lightning_mlp import MultiLayerPerceptron
from lightning_data_module import MnistDataModule
import torch
import pytorch_lightning as pl

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s %(message)s')

def main():
    logging.info("lightning_trainer.main()")

    torch.manual_seed(1)
    mnist_dm = MnistDataModule()

    mnistclassifier = MultiLayerPerceptron()

    """if torch.cuda.is_available():
        trainer = pl.Trainer(max_epochs=10, accelerator='gpu')
    else:
        trainer = pl.Trainer(max_epochs=10)

    trainer.fit(model=mnistclassifier, datamodule=mnist_dm)
    """
    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=10)
    trainer.fit(model=mnistclassifier, datamodule=mnist_dm)

if __name__ == '__main__':
    main()