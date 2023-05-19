from torchvision.transforms import ToTensor
import torch
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import ipywidgets as widgets
import matplotlib

from matplotlib.widgets import Button

from dataset import CartoonImageDataset

matplotlib.use('TkAgg')


dataset = CartoonImageDataset(root_dir="../../../../naver-webtoon-data/faces/", transform=ToTensor())
print(len(dataset))

class Index:
    ind = 0

    def next(self, event):
        self.ind += 1
        if self.ind >= len(dataset):
            self.ind = 0
        self.show_image_pair()

    def prev(self, event):
        self.ind -= 1
        if self.ind < 0:
            self.ind = len(dataset) - 1
        self.show_image_pair()

    def show_image_pair(self):
        src_image, _ = dataset[self.ind]
        src_image = src_image.permute(1, 2, 0)

        src_ax.imshow(src_image)
        fig.canvas.draw()

callback = Index()

fig, (sketch_ax, src_ax) = plt.subplots(1, 2, figsize=(10, 5))
sketch_ax.set_title('Sketch Image')
sketch_ax.axis('off')
src_ax.set_title('Source Image')
src_ax.axis('off')

fig.subplots_adjust(bottom=0.2)
callback.show_image_pair()

axprev = fig.add_axes([0.7, 0.05, 0.1, 0.075])
axnext = fig.add_axes([0.81, 0.05, 0.1, 0.075])
bnext = Button(axnext, 'Next')
bnext.on_clicked(callback.next)
bprev = Button(axprev, 'Previous')
bprev.on_clicked(callback.prev)

plt.show()