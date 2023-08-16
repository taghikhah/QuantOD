import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import random


from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, SVHN
from datasets.lsun import CroppedLSUN, ResizedLSUN
from datasets.cifar import SubsetCIFAR100
from datasets.textures import Textures
from datasets.places import Places365
from datasets.isun import ISUN


class CustomDataset(Dataset):
    def __init__(self, state):
        self.data_path = state['data_path']
        self.batch_size = state['batch_size']
        self.num_workers = state['num_workers']
        self.validation_split = state['validation_split']
        self.id_dataset = state['id_dataset']
        self.ood_dataset = state['ood_dataset']
        self.shuffle = state['shuffle_data']
        self.normalize = state['normalize_data']
        self.image_size = state['image_size']
        self.resize = state['resize_image']
        self.seed = state['seed']
        self.prelims = not state['no_prelims']
        self.get_id_dataset()
        self.get_ood_dataset(self.ood_dataset)

    def __len__(self):
        return len(self.train_set)

    def __getitem__(self, index):
        return self.train_set[index]

    def split_dataset(self, train_data):
        val_offset = int(len(train_data) * (1 - self.validation_split))
        train_data_ = PartialDataset(train_data, 0, val_offset)
        eval_data = PartialDataset(train_data, val_offset, len(train_data) - val_offset)

        return train_data_, eval_data

    def get_transform(self, is_train, is_resize=False):
        transform_list = []
        if is_train:
            if is_resize:
                transform_list.append(T.Resize(self.image_size))
            transform_list.extend([T.RandomHorizontalFlip(), T.RandomCrop(self.image_size, padding=4)])
        
        if is_resize:
            transform_list.extend([T.Resize(self.image_size), T.CenterCrop(self.image_size)])
        
        transform_list.append(T.ToTensor())
        if self.normalize:
            transform_list.append(T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]))

        return T.Compose(transform_list)

    def get_random_classes(self, k):
        random.seed(self.seed)
        return random.sample(range(100), k)

    def get_id_dataset(self):
        train_set, test_set, num_classes = None, None, None
        if self.id_dataset == 'cifar10':
            train_set = CIFAR10(self.data_path, train=True, transform=self.get_transform(True, self.resize), download=True)
            test_set = CIFAR10(self.data_path, train=False, transform=self.get_transform(False, self.resize), download=True)
            num_classes = 10
        elif self.id_dataset == 'cifar100':
            train_set = CIFAR100(self.data_path, train=True, transform=self.get_transform(True, self.resize), download=True)
            test_set = CIFAR100(self.data_path, train=False, transform=self.get_transform(False, self.resize), download=True)
            num_classes = 100
        elif self.id_dataset.startswith('cifar-'):
            k = int(self.id_dataset.split('-')[1]) # the actual number of classes
            train_set = SubsetCIFAR100(self.data_path, train=True, labels=self.get_random_classes(k), transform=self.get_transform(True, self.resize), download=True)
            test_set = SubsetCIFAR100(self.data_path, train=False, labels=self.get_random_classes(k), transform=self.get_transform(False, self.resize), download=True)
            num_classes = 100 # to use the same number of classes as pretrained network on CIFAR-100
        else:
            raise ValueError('Dataset not supported: {}'.format(self.id_dataset))
        # Number of classes
        self.num_classes = num_classes
        # Prelims as Outlier Exposure suggests
        self.num_samples = len(test_set) // 5
        self.num_batches = max(1, self.num_samples // self.batch_size) 
        self.train_set, eval_set = self.split_dataset(train_set)
        # Data Loaders
        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers, pin_memory=False)
        self.eval_loader = DataLoader(eval_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=False)
        self.test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=False)
               
    def get_ood_dataset(self, ood_dataset):
        ood_set = None
        if ood_dataset == 'all':
            ood_datasets = ['textures', 'svhn', 'places365', 'lsunc', 'lsunr', 'isun']
            self.ood_loaders = []
            for ood in ood_datasets:
                ood_loader = self.get_ood_dataset(ood)
                self.ood_loaders.append((ood_loader, ood))
            return
        elif ood_dataset == 'svhn':
            ood_set = SVHN(self.data_path, split='test', transform=self.get_transform(False, self.resize), download=True)
        elif ood_dataset == 'lsunc': 
            ood_set = CroppedLSUN(self.data_path, transform=self.get_transform(False, self.resize), download=True)
        elif ood_dataset == 'lsunr': 
            ood_set = ResizedLSUN(self.data_path, transform=self.get_transform(False, self.resize), download=True)
        elif ood_dataset == 'isun':
            ood_set = ISUN(self.data_path, transform=self.get_transform(False, self.resize), download=True)
        elif ood_dataset == 'places365':
            ood_set = Places365(self.data_path, transform=self.get_transform(False, True), download=True)
        elif ood_dataset == 'textures':
            ood_set = Textures(self.data_path, transform=self.get_transform(False, True), download=True)
        elif ood_dataset == '':
            return 
        else:
            raise ValueError('OOD Dataset not supported: {}'.format(self.ood_dataset))
        
        self.ood_loader = DataLoader(ood_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=False)

        return self.ood_loader

            


class PartialDataset(Dataset):
    def __init__(self, parent_ds, offset, length):
        self.parent_ds = parent_ds
        self.offset = offset
        self.length = length
        assert len(parent_ds) >= offset + length, Exception("Parent Dataset not long enough")
        super(PartialDataset, self).__init__()

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return self.parent_ds[i + self.offset]
