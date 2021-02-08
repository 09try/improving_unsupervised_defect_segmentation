from torchvision import transforms as T
import torch.optim as optim
import torch

from PIL import Image

import os

from utils import display_image
from dataset import MyDataset
from models import Model

root_dir = r'data\grid'

device = (torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu'))
print(device)

img_path = os.path.join(root_dir, 'good', '000.png')
img = Image.open(img_path)
#display_image(img)


# dataset
t = T.Compose([
        T.ToTensor()
])

grid_dataset = MyDataset(os.path.join(root_dir, 'train', 'augmented'), transforms=t)

img = grid_dataset[0]
display_image(img)

# model
model = Model()
out = model(img.unsqueeze(0))
display_image(out.squeeze(0))

# optimizer
learning_rate = 0.001
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# loss function


# training loop