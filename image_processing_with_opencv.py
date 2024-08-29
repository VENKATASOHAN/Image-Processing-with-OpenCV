import pandas as pd
import numpy as np
import cv2

from glob import glob
import matplotlib.pylab as plt

dog_file = glob('build/dogs/*.jpg')
cat_file = glob('build/cats/*.jpg')

"""> Read image with the Matplot lib"""

img_mplt= plt.imread(dog_file[0])
img_mplt

"""> Type"""

type(img_mplt)

"""> Shape"""

img_mplt.shape

"""> Read image with opencv"""

img_cv = cv2.imread(cat_file[20])
img_cv

fig, ax = plt.subplots(figsize=(10,10))
ax.imshow(img_mplt)
plt.show()

fig, ax = plt.subplots(figsize=(10,10))
ax.imshow(img_mplt)
ax.axis('off')
plt.show()

fig, axs=plt.subplots(1,3 ,figsize=(15,5))
axs[0].imshow(img_mplt[:,:,0],cmap='Reds')
axs[1].imshow(img_mplt[:,:,1],cmap='Greens')
axs[2].imshow(img_mplt[:,:,2],cmap='Blues')
axs[0].axis('off')
axs[1].axis('off')
axs[2].axis('off')
plt.show()

fig, axs=plt.subplots(1,2, figsize=(10,5))
axs[0].imshow(img_cv)
axs[1].imshow(img_mplt)
axs[0].set_title("CV Image")
axs[1].set_title("Matplot lib Image")
plt.show()

"""> Image Manipulation"""

img=plt.imread(dog_file[84])
fig, ax = plt.subplots(figsize=(8,8))
ax.imshow(img)
ax.axis('off')
plt.show()

img_gry=cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
img_gry.shape
fig, ax = plt.subplots(figsize=(8,8))
ax.imshow(img_gry, cmap='Greys')
ax.axis('off')
ax.set_title("Grey Image")
plt.show()

"""> Resize and Scaling"""

img_rezie=cv2.resize(img,None, fx=0.25, fy=0.25)
fix , ax = plt.subplots(figsize=(8,8))
ax.imshow(img_rezie)
ax.axis('off')
plt.show()

"""> Resize"""

img_rezie1=cv2.resize(img,(100,100))
fix , ax = plt.subplots(figsize=(8,8))
ax.imshow(img_rezie1)
ax.axis('off')
plt.show()

img_rezie=cv2.resize(img,(5000,5000), interpolation=cv2.INTER_CUBIC)
fix , ax = plt.subplots(figsize=(8,8))
ax.imshow(img_rezie)
ax.axis('off')
plt.show()

img_rezie.shape

"""> Sharpen Image"""

kernel_sharp = np.array([[-1, -1, -1],
                         [-1, 9, -1],
                         [-1, -1, -1]])

sharpened_img = cv2.filter2D(img, -1, kernel_sharp)

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cv2.cvtColor(sharpened_img, cv2.COLOR_BGR2RGB))
ax.axis('off')
plt.show()

"""> Bluring the image"""

kernal_3= np.ones((3,3), np.float32)/ 9
blurred= cv2.filter2D(img,-1, kernal_3)
fix , ax = plt.subplots(figsize=(8,8))
ax.imshow(blurred)
ax.axis('off')
plt.show()

"""> Saving the image"""

plt.imsave('mplt_dog.png', blurred)

cv2.imwrite('cvs_dog.png',sharpened_img)
