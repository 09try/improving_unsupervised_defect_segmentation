import torch
import torchvision.transforms as T
from models import Model

from PIL import Image

import os

from utils import display_image, display_heat_map

device = torch.device('cpu')

model_path = r'models\2021_02_11__21_23_32_model.pt'

model = Model()
model.load_state_dict(torch.load(model_path))
model.eval()

root_dir = r'data\grid'
results_dir = 'results'
test_img_path = os.path.join(root_dir, 'test', '000.png')
test_img = Image.open(test_img_path)

transform = T.Compose([
    T.Resize(256),
    T.RandomCrop(128),
    T.ToTensor()
])
    
test_img_t = transform(test_img)
display_image(test_img_t, save_image=True, name=os.path.join(results_dir, 'in2.png'))
test_img_t = test_img_t.to(device)
test_img_b = test_img_t.unsqueeze(0)
out = model(test_img_b)
out = out.squeeze(0)
display_image(out, save_image=True, name=os.path.join(results_dir, 'out2.png'))
display_heat_map(
    test_img_t.squeeze(0), 
    out.squeeze(0).squeeze(0), 
    save_image=True, 
    name=os.path.join(results_dir, 'heat_map.png'))