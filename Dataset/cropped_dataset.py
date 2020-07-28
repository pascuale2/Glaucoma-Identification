import os
from PIL import Image
from torch.utils.data import Dataset as BaseDataset
from torchvision.transforms import functional as FF

# preprocess the input image
def crop_image_with_percentage(img):
    wid, hei = img.size
    
    # (x1, x2) = 0%, 56.5%
    start_width_percentage = 0
    end_width_percentage = 0.56497175141
    # (y11, y2) = 17%, 75.3%
    start_height_percentage = 0.17023346303
    end_height_percentage = 0.75389105058
    
    new_pts = (round(start_width_percentage*wid), round(start_height_percentage*hei), round(end_width_percentage*wid), round(end_height_percentage*hei))
    return img.crop(new_pts)

def fix_mask(mask):
    mask[mask == 128] = 1
    mask[mask == 255] = 2 
    return mask

class cropped_Dataset(BaseDataset):
  def __init__(self, root_dir):
    self.root_dir = root_dir
    self.images_dir = os.path.join(root_dir, 'Images')
    self.masks_dir = os.path.join(root_dir, 'Masks')
    self.ids = os.listdir(self.images_dir)
    self.maskids = os.listdir(self.masks_dir)

  def __getitem__(self, idx):
    #load images and masks
    img_path = os.path.join(self.images_dir, self.ids[idx])
    mask_path = os.path.join(self.masks_dir, self.maskids[idx])
    original_image = Image.open(img_path).convert("RGB")
    mask = Image.open(mask_path)
    label = 0 if self.ids[idx][0] == 'n' else 1

    cropped_img =  crop_image_with_percentage(original_image)
    mask = crop_image_with_percentage(mask)

    # transform
    cropped_img = cropped_img.resize((256, 256))
    cropped_img = FF.to_tensor(np.array(cropped_img))
    mask = mask.resize((256,256), resample = Image.NEAREST)

    # Fixing mask values 
    mask = np.array(mask)
    mask = np.expand_dims(mask, axis=0)

    mask =  torch.from_numpy(mask)
    mask = fix_mask(mask)
    sample = {'image': cropped_img, 'mask': mask}
    
    return sample
  
  def __len__(self):
    return len(self.ids)
