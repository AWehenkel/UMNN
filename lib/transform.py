import numpy as np
import torch
import torchvision

def logit(x, alpha=1E-6):
    y = alpha + (1.-2*alpha)*x
    return np.log(y) - np.log(1. - y)

def logit_back(x, alpha=1E-6):
    y = torch.sigmoid(x)
    return (y - alpha)/(1.-2*alpha)

class AddUniformNoise(object):
    def __init__(self, alpha=1E-6):
        self.alpha = alpha
    def __call__(self,samples):
        samples = np.array(samples,dtype = np.float32)
        samples += np.random.uniform(size = samples.shape)
        samples = logit(samples/256., self.alpha)
        return samples

class ToTensor(object):
    def __init__(self):
        pass
    def __call__(self,samples):
        samples = torch.from_numpy(samples).float()
        return samples

class ZeroPadding(object):
    def __init__(self,num):
        self.num = num
    def __call__(self,samples):
        samples = np.array(samples,dtype = np.float32)
        tmp = np.zeros((32,32))
        tmp[self.num:samples.shape[0]+self.num,self.num:samples.shape[1]+self.num] = samples
        return tmp

class Crop(object):
    def __init__(self,num):
        self.num = num
    def __call__(self,samples):
        samples = np.array(samples,dtype = np.float32)
        return samples[self.num:-self.num,self.num:-self.num]

class HorizontalFlip(object):
    def __init__(self):
        pass
    def __call__(self,samples):
        return torchvision.transforms.functional.hflip(samples)

class Transpose(object):
    def __init__(self):
        pass
    def __call__(self,samples):
        return np.transpose(samples, (2, 0, 1))

class Resize(object):
    def __init__(self):
        pass
    def __call__(self, samples):
        return torchvision.transforms.functional.resize(samples, [32, 32])
