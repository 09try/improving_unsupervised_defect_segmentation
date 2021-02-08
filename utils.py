import matplotlib.pyplot as plt
import torchvision.transforms as T
import torch

def display_image(image, figsize=(5, 5)):
    plt.figure(figsize=figsize)
    
    if torch.is_tensor(image):
        to_pil_image = T.ToPILImage()
        image = to_pil_image(image)    
    
    plt.imshow(image, cmap='gray')
    plt.show()
    