import os
import gdown
import tarfile

from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


class Textures(Dataset):
    def __init__(self, data_path, transform=None, download=False):
        self.data_path = data_path
        self.transform = transform
        self.dataset_url = 'https://drive.google.com/uc?id=1ZVkKpV3ftfoHDu-zzWzPU1ggI8E69JYh'
        self.file_name = "Textures.tar.gz"
        self.subfolder_name = "dtd/images" 
        
        if download:
            self.download_data()
        if not os.path.exists(f'{self.data_path}/{self.subfolder_name}'):
            print(f"{self.file_name.split('.')[0]} dataset not found! Please download the dataset first.")
            exit()

        self.dataset = ImageFolder(root=f'{self.data_path}/{self.subfolder_name}', transform=transform)
     

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
    
    def download_data(self):
        if not os.path.exists(f'{self.data_path}/{self.subfolder_name}'):
            print(f"\n Downloading {self.file_name.split('.')[0]} dataset from Google Drive ...")
            gdown.download(self.dataset_url, output=os.path.join(self.data_path, self.file_name), quiet=False)
            tar = tarfile.open(os.path.join(self.data_path, self.file_name))
            tar.extractall(self.data_path)
            tar.close()
            print('Download completed! \n')
        else:
            print(f"{self.file_name.split('.')[0]} (DTD) dataset already downloaded and verified: {os.path.join(self.data_path, self.file_name)}")