from torch.utils.data import Dataset, random_split
from torchvision import datasets, transforms

def MnistDataset(data_dir, validation_split):
    trsfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST(data_dir, download=True, transform=trsfm)
    return random_split(
        dataset, [int((1-validation_split)*len(dataset)), int(validation_split*len(dataset))]
    )
