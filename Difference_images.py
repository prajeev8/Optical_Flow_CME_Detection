#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 21:48:17 2024

@author: pallavirajeev
"""

from astropy.io import fits
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt

def open_fits_file(fits_data):
    frames = []
    headers = []
    for filename in sorted(os.listdir(fits_data)):
        if filename.endswith(".fts"):
            print(f"Processing FITS file: {filename}")
            with fits.open(os.path.join(fits_data, filename)) as hdul:
                for hdu in hdul:
                    if hdu.data is not None:
                        mean_value = np.mean(hdu.data)
                        img_normalized = hdu.data / mean_value
                        background = cv2.GaussianBlur(img_normalized, (0,0), 100)
                        subtracted_img = img_normalized - background
                        frames.append(subtracted_img)
                        headers.append(hdu.header)
    return frames, headers

def create_circular_mask(shape, center, radius):
    Y, X = np.ogrid[:shape[0], :shape[1]]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    mask = dist_from_center <= radius
    return mask

def create_difference_imgs(images, headers, date):
    difference_images = []
    for i in range(len(images) - 1):
        img1 = images[i]
        img2 = images[i + 1]
        if img1.shape == img2.shape == (1024, 1024):
            difference_image = img1 - img2
            
            header1 = headers[i]
            center_x = header1['CRPIX1']
            center_y = header1['CRPIX2']
            radius = 180  
            mask = create_circular_mask(difference_image.shape, (center_x, center_y), radius)
            difference_image[mask] = 0
            
            difference_images.append(difference_image)
            
    vmin = np.min([np.min(diff_img)*np.exp(-6) for diff_img in difference_images])
    vmax = np.max([np.max(diff_img)*np.exp(-6) for diff_img in difference_images])
    frames = []
    
    for i, diff_img in enumerate(difference_images):
        fig, ax = plt.subplots(figsize=(11, 11))
        cax = ax.imshow(np.rot90(diff_img), vmin = vmin, vmax = vmax) 
        ax.axis('off')
        os.makedirs(f"{date}_difference_images", exist_ok=True)
        plt.savefig(f"{date}_difference_images/difference_image{i:02d}.png", bbox_inches='tight', pad_inches=0)
        plt.show()
        frames.append(np.array(fig))

fits_data = "folder_with_solar_fits_data" # Give the name of the folder where the data is stored
date = "date_of_the_event" # Type in the date for ease of finding the folder
images, headers = open_fits_file(fits_data)
create_difference_imgs(images, headers, date)









