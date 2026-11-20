from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Images are scaled to [-1, 1] to match the generator's final Tanh.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])


def get_mnist(root="data", train=True):
    return datasets.MNIST(root=root, train=train, download=True, transform=transform)


def get_loader(batch_size=128, root="data", train=True, shuffle=True, generator=None):
    return DataLoader(
        get_mnist(root, train),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        generator=generator,
    )
