from torchvision import transforms as T
import torch.optim as optim
import torch

import matplotlib.pyplot as plt

from PIL import Image

import os

from utils import display_image, display_heat_map
from dataset import MyDataset
from models import Model, init_weights
from losses import SSIMLoss

import datetime

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
loss_fn = SSIMLoss(device)
loss_fn = loss_fn.to(device)

# training loop
n_epochs = 1000
def training_loop(model, optimizer, loss_fn, dataloader, n_epochs):
    model.to(device)
    model.train()
    start = datetime.datetime.now()
    for epoch in range(n_epochs):
        loss_train = 0
        for images in dataloader:
            
            images = images.to(device)
            outputs = model(images)
            
            loss = loss_fn(images, outputs)
            loss_train += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        if epoch == 0 or epoch % 5 == 0:
            l = loss_train / images.shape[0]
            model.losses.append(l)
            print('epoch {} loss {:.4f} {}'.format(epoch, l, datetime.datetime.now().strftime('%H:%M:%S')))
    end = datetime.datetime.now()
    diff = end - start
    print('elapsed', diff)
         
    
model.apply(init_weights)

dataloader = torch.utils.data.DataLoader(grid_dataset, batch_size=128, shuffle=True)
training_loop(model, optimizer, loss_fn, dataloader, n_epochs)

plt.plot(model.losses)

test_img_path = os.path.join(root_dir, 'test', '000.png')
test_img = Image.open(test_img_path)

transform = T.Compose([
    T.Resize(256),
    T.RandomCrop(128),
    T.ToTensor()
])
    
test_img_t = transform(test_img)
display_image(test_img_t)
test_img_t = test_img_t.to(device)
test_img_b = test_img_t.unsqueeze(0)
print(test_img_b.shape)
    
model.eval()
out = model(test_img_b)
display_image(out.squeeze(0))
display_heat_map(test_img_t.squeeze(0), out.squeeze(0).squeeze(0))