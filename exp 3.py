#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt


# In[5]:


gray_image = cv2.imread('Car image.jpg', cv2.IMREAD_GRAYSCALE)


# In[6]:


plt.title("Car image")
plt.imshow(gray_image, cmap='gray')
plt.axis('off')


# In[9]:


plt.title("Histogram of Grayscale Image")
plt.hist(gray_image.ravel(), bins=256, color='black', alpha=0.6)
plt.xlim(0, 255)
plt.tight_layout()
plt.show()


# In[8]:


equalized_gray_image = cv2.equalizeHist(gray_image)


# In[10]:


plt.title("Histogram of Equalized Car Image")
plt.hist(equalized_gray_image.ravel(), bins=256, color='black', alpha=0.6)
plt.xlim(0, 255)


# In[11]:


equalized_gray_image = cv2.equalizeHist(gray_image)


# In[12]:


plt.title("Histogram of Equalized Grayscale Image")
plt.hist(equalized_gray_image.ravel(), bins=256, color='black', alpha=0.6)
plt.xlim(0, 255)


# In[13]:


plt.title("Enhanced Grayscale Image")
plt.imshow(equalized_gray_image, cmap='gray')
plt.axis('off')


# In[19]:


import cv2
import numpy as np
import matplotlib.pyplot as plt


# In[20]:


color_image = cv2.imread('bird.png')


# In[21]:


plt.title("Input Color Image")
plt.imshow(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
plt.axis('off')


# In[22]:


hist_b = cv2.calcHist([color_image], [0], None, [256], [0, 256])
hist_g = cv2.calcHist([color_image], [1], None, [256], [0, 256])
hist_r = cv2.calcHist([color_image], [2], None, [256], [0, 256])


# In[23]:


plt.title("Histogram of Input Color Image")
plt.plot(hist_b, color='blue', label='Blue channel')
plt.plot(hist_g, color='green', label='Green channel')
plt.plot(hist_r, color='red', label='Red channel')
plt.show()


# In[24]:


blue_channel_eq = cv2.equalizeHist(color_image[:, :, 0])
green_channel_eq = cv2.equalizeHist(color_image[:, :, 1])
red_channel_eq = cv2.equalizeHist(color_image[:, :, 2])


# In[25]:


equalized_color_image = cv2.merge([blue_channel_eq, green_channel_eq, red_channel_eq])


# In[ ]:




