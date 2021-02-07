from PIL import Image
import matplotlib.pyplot as plt

import os

root_dir = r'G:\source\repos\improving_unsupervised_defect_segmentation\data\grid'

img_path = os.path.join(root_dir, 'good', '000.png')
img = Image.open(img_path)
plt.imshow(img)
plt.show()