import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import sseg_dataset100 as dataset
from torchvision import transforms
from utils import split_train_and_validation
from torch.utils.data import DataLoader, SubsetRandomSampler

def inspect_dataset(dataset):
    for i in range(len(dataset)):
        print(i)

        image = dataset[i]['img_data']
        mask_raw = dataset[i]['mask_raw']
        mask = dataset[i]['mask']

        print('\timage', dataset[i]['img_data'].shape)
        print('\tmask_raw', dataset[i]['mask_raw'].shape)    
        print('\tmask', dataset[i]['mask'].shape)
        print('\tpixels', set(list(mask_raw.numpy().flatten())))
        print('\tclasses', set(list(mask.numpy().flatten())))

def inspect_dataloader(dataloader):
    for batch, sample in enumerate(dataloader):
        print('Batch {}'.format(batch))
        print('\timage', sample['img_data'].shape)
        print('\tmask_raw', sample['mask_raw'].shape)
        print('\tmask', sample['mask'].shape)

if __name__ == '__main__':
    folder = '/Users/faisal/Google Drive File Stream/My Drive/Datasets/a-benchmark-for-semantic-image-segmentation/Semantic dataset100'
    transform = transforms.Compose([transforms.ToTensor()])

    dataset = dataset.SemanticSegmentationDataset100(folder=folder, size=(256,256), transform=transform)

    #inspect_dataset(dataset)

    train_indices, validation_indices = split_train_and_validation(dataset)
    train_sampler = SubsetRandomSampler(train_indices)
    validation_sampler = SubsetRandomSampler(validation_indices)

    train_loader = DataLoader(dataset, batch_size=8, sampler=train_sampler)
    validation_loader = DataLoader(dataset, batch_size=8, sampler=validation_sampler)

    #inspect_dataloader(train_loader)
    #inspect_dataloader(validation_loader)

