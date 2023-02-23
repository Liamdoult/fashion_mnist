import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as T



class FashionMnistImageTensor:

  def __init__(self, name: str, path: str, classification: int) -> None:
    self.name = name
    self.path = path
    self.classification = classification
    self.raw_img_tensor = torchvision.io.read_image(f"{self.path}/{name}")
    self.img_tensor = (torch.flatten(self.raw_img_tensor, start_dim=1, end_dim=-1)-127.5)/127.5

  def draw(self):
    FashionMnistImageTensor.draw_many([self.raw_img_tensor])

  def draw_many(imgs):
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
      img = T.ToPILImage()(img.to('cpu'))
      axs[0, i].imshow(np.asarray(img), cmap='gray')
      axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])