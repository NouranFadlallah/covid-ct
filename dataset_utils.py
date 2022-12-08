from torch.autograd import Variable
from PIL import Image
import cv2

def image_loader(image_name):
    #load image, returns cuda tensor
    image = Image.open(image_name).convert('RGB')
    image = loader(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)
    return image.to(device, non_blocking=True)

def image_from_array_loader(img_arr):
    im_rgb = cv2.cvtColor(img_arr, cv2.COLOR_GRAY2RGB)
    image = Image.fromarray(img_arr, 'RGB')
    image = loader(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)
    return image.to(device, non_blocking=True)
