import torch
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class AttributeDataset(Dataset):

    def __init__(self, root, transform=None, isEmbed=True,isAge=True):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.transform = transform
        self.isEmbed = isEmbed
        self.isAge = isAge

        if not self.isEmbed:
          self.training_files_names = []
          self.root_dir = root
          images=[img.name for img in os.scandir(root)]
          for img in images:
              self.training_files_names.append(os.path.join(root,img))
        else:
          self.dataset=torch.load(root+"/dataset.pt")
          


    def __len__(self):
        
        if self.isEmbed:
          return len(self.dataset)
        else:
          return len(self.training_files_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.isEmbed:
          dataset=self.dataset.data.cpu().numpy()
          features=dataset[idx,:-1]
          label=int(dataset[idx,-1])

        else:
          features = np.asarray(Image.open(self.training_files_names[idx])) / 255.0
          features = np.transpose(features.astype(np.double))

          if self.isAge:
            label = float(self.training_files_names[idx].split("/")[-1][:].split("_")[0]) #age
          else:
            label = int(self.training_files_names[idx].split("/")[-1][:].split("_")[1]) #gender

        return features, label

