from torch.utils.data import Dataset, random_split
from torchvision import datasets, transforms

trsfm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def MnistDataset(data_dir, validation_split):
    dataset = datasets.MNIST(data_dir, download=True, transform=trsfm)
    num_trainset = int((1-validation_split)*len(dataset))
    num_validset = len(dataset) - num_trainset
    return random_split(
        dataset, [num_trainset, num_validset]
    )

def MnistTestDataset(data_dir):
    dataset = datasets.MNIST(data_dir, train=False, download=True, transform=trsfm)
    return dataset