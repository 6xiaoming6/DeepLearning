import torch
from PIL import Image


def load_image_constant_size(image_path, size=(256, 256)):
    img = Image.open(image_path)
    temp = max(img.size)
    mask = Image.new('RGB', (temp, temp), (0, 0, 0))
    mask.paste(img, (0, 0))
    mask = mask.resize(size=size)
    return mask