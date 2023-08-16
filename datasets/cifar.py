import numpy as np

from torchvision.datasets import CIFAR100

class SubsetCIFAR100(CIFAR100):
    def __init__(self, root, labels, train, transform, download):
        super().__init__(root=root, train=train, transform=transform, download=download)
        
        # Create a mask for the desired labels
        mask = np.isin(self.targets, labels)
        
        # Select only the data entries with the desired labels
        self.data = self.data[mask]
        self.targets = np.array(self.targets)[mask].tolist()

        # Remap the selected labels to start at 0 and increase consecutively
        self.labels_map = {label: idx for idx, label in enumerate(sorted(set(self.targets)))}
        self.targets = [self.labels_map[label] for label in self.targets]
        
        # Reassign classes and class_to_idx
        self.classes = [self.classes[i] for i in labels]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}