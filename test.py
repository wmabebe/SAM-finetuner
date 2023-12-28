from pycocotools.coco import COCO
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
from dataloaders import CocoDetection, MyCocoDataset
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

# In my case, just added ToTensor
def get_transform():
    custom_transforms = []
    custom_transforms.append(transforms.ToTensor())
    return transforms.Compose(custom_transforms)

# collate_fn needs for batch
def collate_fn(batch):
    return tuple(zip(*batch))


# path to your own data and coco file
data_dir = 'datasets/coco/val2017'
ann_file = 'datasets/coco/annotations/instances_val2017.json'


img_transform = transforms.Compose([transforms.ToTensor()])

coco_dataset  = CocoDetection(data_dir,ann_file,get_transform())

print(coco_dataset)

# Create a DataLoader with batch size 8
batch_size = 1
data_loader = DataLoader(dataset=coco_dataset, batch_size=batch_size, shuffle=True)

# Iterate through the DataLoader
for idx, (images, targets) in enumerate(data_loader):
    # Access batch elements (features, labels, etc.)
    print(f'{idx} images:{images.shape}, target:{targets}')
    if idx == 9:
        break

# dataDir='datasets/coco'
# dataType='val2017'
# annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)

# coco = COCO(annFile)

# coco.info()

# # display COCO categories and supercategories
# cats = coco.loadCats(coco.getCatIds())
# nms=[cat['name'] for cat in cats]
# print('COCO categories: \n{}\n'.format(' '.join(nms)))

# nms = set([cat['supercategory'] for cat in cats])
# print('COCO supercategories: \n{}'.format(' '.join(nms)))

# # get all images containing given categories, select one at random
# catIds = coco.getCatIds(catNms=['person','dog','skateboard']);
# imgIds = coco.getImgIds(catIds=catIds );
# imgIds = coco.getImgIds(imgIds = [324158])
# img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]

# # load and display image
# # I = io.imread('%s/images/%s/%s'%(dataDir,dataType,img['file_name']))
# # use url to load image
# I = io.imread(img['coco_url'])
# plt.axis('off')
# plt.imshow(I)
# plt.savefig("test.jpg")

# plt.imshow(I); plt.axis('off')
# annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
# anns = coco.loadAnns(annIds)
# coco.showAnns(anns)

