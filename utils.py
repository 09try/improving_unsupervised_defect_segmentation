import matplotlib.pyplot as plt
import torchvision.transforms as T
import torch
from skimage.measure import compare_ssim as ssim

def display_image(image, figsize=(5, 5)):
    plt.figure(figsize=figsize)
    
    if torch.is_tensor(image):
        to_pil_image = T.ToPILImage()
        image = to_pil_image(image)    
    
    plt.imshow(image, cmap='gray')
    plt.show()
    
    
def display_heat_map(image1, image2):
    print(image1.shape)
    print(image2.shape)
    _, _, S = ssim(image1.detach().cpu().numpy()[1:-1, 1:-1], image2.detach().cpu().numpy()[1:-1, 1:-1], gradient=True, full=True, multichannel=False)
    plt.imshow(1-S, vmax=1, cmap="jet")
    plt.colorbar()