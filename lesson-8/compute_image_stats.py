#
# The goal is to compute per channel mean and variances for images present in a folder

import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import argparse

def mk_filelist(folder, extension):
    files = []
    filelist = os.listdir(folder)
    for file in filelist[:]:
        if file.endswith(extension):
            files.append(file)
    return sorted(files), folder

class Accum:
    """Use this class to compute per channel mean and variance for torch tensors of the form (nchannels x height x rows)."""

    def __init__(self):
        self.first = True

    def add(self, im):
        dim = im.shape
        im = im.view(im.shape[0],-1)
        self.npixels = im.shape[1]
        x = torch.sum(im, 1, True)
        x2 = torch.sum(torch.pow(im, 2), 1, True)

        if self.first:
            print('Image dimensions:\n\t', dim)
            self.first = False
            self.s = x
            self.s2 = x2
            self.n = 1
        else:
            self.s = torch.add(self.s, x)
            self.s2 = torch.add(self.s2, x2)
            self.n += 1
        
    def mean(self):
        return self.s / (self.npixels * self.n)

    def var(self):
        t = self.n * self.npixels
        return (self.s2 - torch.pow(self.s, 2) / t) / t 

    def stdev(self):
        return torch.sqrt(self.var())

def compute_stats(folder, files, transform=None):
    if not transform:
        transform = transforms.ToTensor() 

    acc = Accum()
    for filename in files:
        filepath = os.path.join(folder, filename)
        image = Image.open(filepath)
        im = transform(image)
        acc.add(im)
    
    return acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute per channel mean and standard deviation for images present in a folder.')
    parser.add_argument('folder', action='store', default='.', help='Folder containing image files')
    parser.add_argument('ext', action='store', default='.', help='Image files extension')
    args = parser.parse_args()

    files = mk_filelist(args.folder, args.ext)
    nfiles = len(files[0])
    print('Found {} images\n'.format(nfiles))

    acc = compute_stats(args.folder, files[0])

    print('Mean:\n\t', acc.mean().squeeze())
    print('Standard deviation:\n\t', acc.stdev().squeeze())

    print('\nComputing mean and standard deviation of whitened data.')
    t = transforms.Compose([transforms.ToTensor(), transforms.Normalize(acc.mean(), acc.stdev())])
    acc2 = compute_stats(args.folder, files[0], t)
    print('Mean (0):\n\t', acc2.mean().squeeze())
    print('Standard deviation (1):\n\t', acc2.stdev().squeeze())

    image_stats = { 'folder': args.folder, 'extension': args.ext, 'mean': acc.mean(), 'var': acc.var(), 'stdev': acc.stdev() }
    print('Saving in {}'.format('image_stats.pt'))
    torch.save(image_stats, 'image_stats.pt')