import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

class MNIST_dataloader():
    def __init__(self, batch_size, train=True):
        # MNIST dataset
        self.batch_size = batch_size
        if train:
            self.dataset = torchvision.datasets.MNIST(root='../../../dataset/',
                                                   train=True,
                                                   transform=transforms.ToTensor(),
                                                   download=False)
        else:

            self.dataset = torchvision.datasets.MNIST(root='../../../dataset/',
                                                  train=False,
                                                  transform=transforms.ToTensor())
        self.dataloader = torch.utils.data.DataLoader(dataset=self.dataset, batch_size=self.batch_size,
                                                      shuffle=True)

def main():
    batch_size = 100
    train_dataloader = MNIST_dataloader(batch_size, train=True)
    for i, (images, labels) in enumerate(train_dataloader.dataloader):
        print(i, images.shape, labels.shape)

# if __name__ == "__main__":
#     main()