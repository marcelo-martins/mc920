import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import warnings
from argparse import RawTextHelpFormatter

warnings.simplefilter("ignore")

def get_parser():
    parser = argparse.ArgumentParser(description='Global and local filters testing', formatter_class=RawTextHelpFormatter)
    parser.add_argument('--folder', default = 'imagens/',
                        help='Folder where the image(s) are')#required = False é um argumento
    parser.add_argument('--option', type=int, choices = range(1,10), default = -1,
                        help = 'Choose option')
    parser.add_argument('--image', default = "baboon", help = 'Choose name of the PGM image you want to run')

    arguments = parser.parse_args()
    return arguments

def plotAndSave(img, nome = "NaN", func = "NaF"):
    cv2.imwrite(f"{func}{nome}.png", img)

def grayscales(img, nome="NaN"):
    final = img.copy() 

    B, G, R = cv2.split(final)
    final = 0.2989* R + 0.5870* G + 0.1140* B 
    plotAndSave(final, nome, "grayscale_")

def edge_detection(img, nome="NaN"):
    final = img.copy()
    edges = cv2.Canny(final,100,200)
    edges = cv2.bitwise_not(edges)
    plotAndSave(edges, nome, "edges_")

if __name__ == '__main__':
    #Getting arguments and reading image
    arguments = get_parser()
    opt = int(arguments.option)
    folder = arguments.folder
    name = arguments.image
    if ".png" not in name:
        name+=".png"

    img = cv2.imread(folder + name, cv2.IMREAD_COLOR)

    
    name = name[:-4] #remove ".png"

    if(opt == -1):
        print("Please insert an option as --option opt")
    if(opt==1):
        grayscales(img, name)
    if(opt==2):
        edge_detection(img, name)
    
        