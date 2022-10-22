import numpy as np
from scipy import ndimage
from skimage.measure import label

data = np.load('ps.npy')
data[data > 0] = 1
labeled = label(data)

print('Number of objects in image:',labeled.max())

print('Number of different objects:')

mask = np.ones((4, 6))
rect = label(ndimage.binary_erosion(data, mask)).max()
print(rect)

mask[2:4, 2:4] = 0
print(label(ndimage.binary_erosion(data, mask)).max() - rect)

mask = np.ones((4, 6))
mask[2:4, 2:4] = 1
mask[0:2, 2:4] = 0
print(label(ndimage.binary_erosion(data, mask)).max() - rect)

mask = np.ones((6,4))
mask[2:4, 2:] = 0
print(label(ndimage.binary_erosion(data, mask)).max())

mask[2:4, 2:] = 1
mask[2:4, 0:2] = 0
print(label(ndimage.binary_erosion(data, mask)).max())