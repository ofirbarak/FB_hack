from vgg_17_keras import *
import numpy as np
from oxford_classes import classes
import matplotlib.pyplot as plt
import cv2

def get_building_name(img_path):
    img = cv2.imread(img_path)
    model = load_model_and_weights('vgg17_demo_weights.h5')
    print(run(model, img))

# print(classes[0])
get_building_name('oxbuild_images/all_souls_000000.jpg')