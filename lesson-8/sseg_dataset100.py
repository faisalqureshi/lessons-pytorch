import os
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms
from PIL import Image

#wget -r --no-parent -nH -R '*\?C=*' --cut-dirs=1 http://vclab.science.uoit.ca/datasets/sseg-dataset100

def mk_filelist(folder, extension):
    files = []
    filelist = os.listdir(folder)
    for file in filelist[:]:
        if file.endswith(extension):
            files.append(file)
    return sorted(files), folder

def sanitize(data):
    # Just to confirm that the tuple filenames match
    for k in data[:]:
        if os.path.splitext(k[0])[0] != os.path.splitext(k[1])[0]:
            print('Files {} and {} don\'t match. IGNORING!'.format(k[0], k[1]))
            data.remove(k)
    return data

class SemanticSegmentationDataset100(Dataset):
    """Constructs SemanticSegmentationDataset100 from images available
    in a folder.
    
    This assumes that images are 'folder/image/*.jpg' and masks are
    'folder/ground-truth/*.png'.
    
    Images and masks are matched using their filenames."""

    def __init__(self, folder, transform, size=(256,256), verbose=False):
        super().__init__()

        images, self.image_folder = mk_filelist(folder=os.path.join(folder,'image'), extension='.jpg')
        masks, self.masks_folder = mk_filelist(folder=os.path.join(folder,'ground-truth'), extension='.png')
        self.filelist = sanitize(list(zip(images, masks)))
        self.num_images = len(self.filelist)
        self.data = []

        print('Found {} images'.format(self.num_images))
        if verbose:
            print('Image folder {}'.format(self.image_folder))
            print('Masks folder {}'.format(self.masks_folder))

        # Unique labels found in the dataset
        unique = set()
        accum = torch.zeros([3,256,256])

        for i,m in self.filelist:
            if verbose: print('Loading image {} and mask {}'.format(i, m))
            # Images are in jpg format
            image = Image.open(os.path.join(self.image_folder, i))
            image = transform(image.resize(size))
            accum.add_(image)
            # Masks are in png format
            mask_raw = Image.open(os.path.join(self.masks_folder, m))
            mask_raw = mask_raw.resize(size)
            # I noticed that for this dataset, the pixels are 
            # not labelled using in terms of classes.  We no create our 
            # own masks where each pixel will contain the correct
            # class labels. 
            mask = (np.array(mask_raw)*255.).astype(int)
            unique = unique.union(mask.flatten())
            self.data.append(
                {
                    'img_data': image,
                    'mask_raw': transform(mask_raw),
                    'mask': mask
                }
            )

        self.sums = torch.sum(accum, dim=(1,2)) / (len(self.data)*size[0]*size[1])
 
        print('Num. of classes', len(unique))
        labels = range(len(unique))
        mapping = list(zip(unique, labels))
        for i in range(len(self.data)):
            mask = self.data[i]['mask']
            for j in mapping:
                mask[mask == j[0]] = j[1]
            self.data[i]['mask'] = torch.tensor(mask.astype(int)).unsqueeze(0)

    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return self.num_images