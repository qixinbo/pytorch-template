from torch.utils.data import Dataset, random_split
from torchvision import datasets, transforms

def MnistDataset(data_dir):
    trsfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST(data_dir, download=True, transform=trsfm)
    train_dataset, validation_dataset, test_dataset = random_split(
        dataset, [50000, 5000, 5000]
    )
    return train_dataset, validation_dataset


def MnistTestDataset(data_dir):
    trsfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST(data_dir, download=True, transform=trsfm)
    train_dataset, validation_dataset, test_dataset = random_split(
        dataset, [50000, 5000, 5000]
    )
    return test_dataset