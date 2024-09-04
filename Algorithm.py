#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 12:05:14 2024

@author: pallavirajeev
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
from astropy.io import fits



def load_frames_from_folder(folder_path):
    file_paths = sorted(glob(os.path.join(folder_path, '*.png')))
    frames = []
    for file_path in file_paths:
        frame = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if frame is not None:
            frames.append(frame)
    return frames

def reduce_noise(img):
    denoised = cv2.fastNlMeansDenoising(img, None, 30, 7, 21)
    denoised = cv2.medianBlur(denoised, 5)
    return denoised

def set_ROI(frame, x_value, y_value, width, height):
    roi = frame[y_value:y_value + height, x_value:x_value + width]
    return roi

def compute_optical_flow_and_magnitude(frames, x_value, y_value, width, height):
    magnitudes = []
    u_list = []
    v_list = []

    prev_frame = set_ROI(frames[0], x_value, y_value, width, height)
    for i in range(1, len(frames)):
        next_frame = set_ROI(frames[i], x_value, y_value, width, height)

        flow = cv2.calcOpticalFlowFarneback(prev_frame, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        u = flow[..., 0]
        v = flow[..., 1]

        velocity = np.sqrt(u**2 + v**2)
        
        D = 8627.5
        F = 12*60
        magnitude = (velocity*D)/F
        magnitudes.append(magnitude)
        u_list.append(u)
        v_list.append(v)

        prev_frame = next_frame
    
    return magnitudes, u_list, v_list

def plot_velocity_heatmap_with_quiver(magnitude, u, v, folder_name, step=15, frame_number=0):
    plt.figure(figsize=(10, 10))
    plt.imshow(magnitude, cmap='hot')
    plt.colorbar(label='Velocity Magnitude (km/s)')
    plt.clim(100)
    plt.title('Velocity Magnitude of CME')
    plt.xlabel('Distance along X-Axis (km)')
    plt.ylabel('Distance along Y-axis (km)')
    plt.gca().invert_yaxis()
    num_ticks = 10  
    tick_labels = np.linspace(0, 8858624, num_ticks)
    plt.xticks(np.linspace(0, magnitude.shape[1] - 1, num_ticks), labels=np.round(tick_labels).astype(int))
    plt.yticks(np.linspace(0, magnitude.shape[0] - 1, num_ticks), labels=np.round(tick_labels).astype(int))

    y, x = np.mgrid[0:magnitude.shape[0]:step, 0:magnitude.shape[1]:step]
    plt.quiver(x, y, u[::step, ::step], v[::step, ::step], color='white', scale=200)

    os.makedirs(f"{folder_name} intensity", exist_ok=True)
    plt.savefig(f"{folder_name} intensity/frame_{frame_number:02d}.png")
    plt.show()

def plot_velocity_vectors_on_original_frames(frames, magnitudes, u_list, v_list, folder_name, step=10, x_value=0, y_value=0):
    os.makedirs(f"frames_with_vectors_{folder_name}", exist_ok=True)
    
    for i, (frame, magnitude, u, v) in enumerate(zip(frames, magnitudes, u_list, v_list)):

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        roi = set_ROI(frame_rgb, x_value, y_value, magnitude.shape[1], magnitude.shape[0])

        y, x = np.mgrid[0:magnitude.shape[0]:step, 0:magnitude.shape[1]:step]
        norm = plt.Normalize(vmin=np.min(magnitude), vmax=np.max(magnitude))
        colormap = plt.cm.get_cmap('hot')

        for j in range(0, x.shape[0]):
            for k in range(0, x.shape[1]):
                magnitude_value = magnitude[y[j, k], x[j, k]]
                color = colormap(norm(magnitude_value))  
                color = tuple(int(255 * c) for c in color[:3])

                cv2.arrowedLine(roi, (x[j, k], y[j, k]), 
                                (x[j, k] + int(u[y[j, k], x[j, k]]), y[j, k] + int(v[y[j, k], x[j, k]])), 
                                color, 1, tipLength=0.4)

        plt.figure(figsize=(10, 10))
        plt.imshow(frame_rgb)
        plt.xlabel('Distance along X-Axis (km)')
        plt.ylabel('Distance along Y-Axis (km)')
        plt.gca().invert_yaxis()
        num_ticks = 10  
        tick_labels = np.linspace(0, 8858624, num_ticks)
        plt.xticks(np.linspace(0, magnitude.shape[1] - 1, num_ticks), labels=np.round(tick_labels).astype(int))
        plt.yticks(np.linspace(0, magnitude.shape[0] - 1, num_ticks), labels=np.round(tick_labels).astype(int))
        plt.savefig(f"frames_with_vectors_{folder_name}/frame_{i:02d}.png")
        plt.close()


def main():

    frames_folder = "21 sept difference images" # Give the name of the folder with difference images. Usually, it is saved as the date_difference images.
    folder_name = "21st sept" # For ease of finding, provide the date 
    noisy_frames = load_frames_from_folder(frames_folder)
    denoised_frames = [reduce_noise(frame) for frame in noisy_frames]
    

    x_value = 0
    y_value = 0
    width = 1024
    height = 1024

    magnitudes, u_list, v_list = compute_optical_flow_and_magnitude(denoised_frames, x_value, y_value, width, height)

    for i, (magnitude, u, v) in enumerate(zip(magnitudes, u_list, v_list)):
        plot_velocity_heatmap_with_quiver(magnitude, u, v, step=15, frame_number=i)
    plot_velocity_vectors_on_original_frames(noisy_frames, magnitudes, u_list, v_list, step=10, x_value=x_value, y_value=y_value)
    

if __name__ == "__main__":
    main()

