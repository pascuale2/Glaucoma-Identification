import os
from PIL import Image
from torch.utils.data import Dataset as BaseDataset

class Dataset(BaseDataset):
  def __init__(self, root_dir, transform=None):
    self.root_dir = root_dir
    self.images_dir = os.path.join(root_dir, 'Images')
    self.masks_dir =os.path.join(root_dir, 'Masks')
    self.transform = transform
    self.ids = os.listdir(self.images_dir)
    self.maskids = os.listdir(self.masks_dir)

  def __getitem__(self, idx):
    #load images and masks
    img_path = os.path.join(self.images_dir, self.ids[idx])
    mask_path = os.path.join(self.masks_dir, self.maskids[idx])
    img = Image.open(img_path).convert("RGB")
    mask = Image.open(mask_path)
    label = 0 if self.ids[idx][0] == 'n' else 1

    sample = {'image': img, 'mask': mask}
    
    if self.transform is not None:
      sample = self.transform(sample)
    return sample
  
  def __len__(self):
    return len(self.ids)
