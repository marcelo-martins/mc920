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

def binaryImage(img, nome="NaN"):
    final = img.copy() 

    final = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)
  
    (thresh, final) = cv2.threshold(final, 254, 255, cv2.THRESH_BINARY)
    plotAndSave(final, nome, "binary_")

def grayscales(img, nome="NaN"):
    final = img.copy() 

    B, G, R = cv2.split(final)
    final = 0.2989* R + 0.5870* G + 0.1140* B  
    plotAndSave(final, nome, "grayscale_")

def edge_detection(img, nome="NaN"):
    final = img.copy()
    v = np.median(final)
    lower = int(max(0, (1.0 - 0.33) * v))
    upper = int(min(255, (1.0 + 0.33) * v))
    edges = cv2.Canny(final, lower, upper)
    #edges = cv2.Canny(final,100,200)
    plotAndSave(cv2.bitwise_not(edges), nome, "edges_")

    return edges

def contour_properties(img, nome="NaN"):
    edges = edge_detection(img, nome)
    contours, h = cv2.findContours(edges, 0, 1)
    
    print("Número de regiões: " + str(len(contours)) + "\n")
    i=0
    for cont in contours:
        area = cv2.contourArea(cont)
        perimeter = cv2.arcLength(cont, True)
        hull = cv2.convexHull(cont)
        hull_area = cv2.contourArea(hull)
        solidity = float(area)/hull_area
        print("Região " + str(i) + ": Área: " + str(format(round(area), "4.0f")) + " Perímetro:  " + str(format(perimeter, "3.6f")) + "  Solidez: " + str(format(solidity, "3.6f")))
        i+=1
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
        binaryImage(img, name)
    if(opt==2):
        grayscales(img, name)
    if(opt==3):
        edge_detection(img, name)
    if(opt==4):
        contour_properties(img, name)
    
        