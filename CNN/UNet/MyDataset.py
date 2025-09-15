import os
from torch.utils.data import Dataset
from torchvision import transforms

from utils import load_image_constant_size

class MyDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.imag_name_list = os.listdir(os.path.join(self.data_path, 'SegmentationClass'))
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.imag_name_list)


    def __getitem__(self, index):
        segment_name = self.imag_name_list[index]
        image_name = segment_name.replace('png', 'jpg')

        image_path = os.path.join(self.data_path, 'JPEGImages', image_name)
        segment_path = os.path.join(self.data_path, 'SegmentationClass', segment_name)

        image, segment = load_image_constant_size(image_path), load_image_constant_size(segment_path)

        return self.transform(image), self.transform(segment)

if __name__ == '__main__':
    dataset = MyDataset(r'/CNN/UNet/dataset/VOC2012')
    print(len(dataset))