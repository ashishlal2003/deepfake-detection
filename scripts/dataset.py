import os
from torch.utils.data import Dataset 
from torchvision import transforms
from PIL import Image

class DeepfakeDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Label 0 for real, 1 for fake
        for label, folder in enumerate(['real', 'fake']):
            folder_path = os.path.join(data_dir, folder)
            for img_file in os.listdir(folder_path):
                self.image_paths.append(os.path.join(folder_path, img_file))
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])