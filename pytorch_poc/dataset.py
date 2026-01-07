import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class MCTextureDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_files = []
        # Filter for 16x16 images only
        for f in os.listdir(root_dir):
            if f.endswith('.png'):
                try:
                    with Image.open(os.path.join(root_dir, f)) as img:
                        if img.size == (16, 16):
                            self.image_files.append(f)
                except:
                    pass
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB') # Convert to RGB to ensure 3 channels
        return self.transform(image)

    def get_images_by_keyword(self, keyword):
        """Returns a list of tensors for images whose filename contains the keyword."""
        tensors = []
        for idx, fname in enumerate(self.image_files):
            if keyword in fname:
                tensors.append(self.__getitem__(idx))
        return tensors

    def get_image_by_name(self, name):
        """Returns a single tensor for a specific filename."""
        for idx, fname in enumerate(self.image_files):
            if name == fname or name in fname: # allow partial match if unique enough
                 return self.__getitem__(idx)
        return None
