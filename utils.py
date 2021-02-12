import matplotlib.pyplot as plt
import torchvision.transforms as T
import torch
from skimage.metrics import structural_similarity

def display_image(image, figsize=(3, 3), save_image=False, name=None):
    plt.figure(figsize=figsize)
    
    if torch.is_tensor(image):
        to_pil_image = T.ToPILImage()
        image = to_pil_image(image)    
    
    plt.imshow(image, cmap='gray')
    if save_image == True:
        plt.savefig(name)
    plt.show()
    
    
def display_heat_map(image1, image2, save_image=False, name=None):
    _, _, S = structural_similarity(image1.detach().cpu().numpy()[1:-1, 1:-1], image2.detach().cpu().numpy()[1:-1, 1:-1], gradient=True, full=True, multichannel=False)
    heat_map = 1-S
   
    plt.imshow(heat_map, vmax=1, cmap="jet")
    plt.colorbar()
    if save_image == True:
        plt.savefig(name)
    plt.show()