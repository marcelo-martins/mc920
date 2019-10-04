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
    # parser.add_argument('--option', type=int, choices = range(1,10), default = -1,
    #                     help = 'Choose filter you want to run\n1 : global\n2 : bernsen\n3 : niblack\n4 : sauvola\n'
    #                         + '5 : sabale\n6 : contrast\n7 : mean\n8 : median\n')
    parser.add_argument('--image', default = "baboon", help = 'Choose name of the PGM image you want to run')

    arguments = parser.parse_args()
    return arguments

def plotAndSave(img):
    cv2.imwrite("nome.png", final) #change format from to pgm to png

def globalMethod(img, nome="nome"):
    final = img.copy() 

    for y in range(0, img.shape[0]):
        for x in range(0, img.shape[1]):
            if(img[y, x] != 255):
                final[y,x] = 0

    return final

if __name__ == '__main__':
    #Getting arguments and reading image
    # arguments = get_parser()
    # opt = int(arguments.option)
    # mask = int(arguments.mask)
    # folder = arguments.folder
    # name = arguments.image
    # if ".png" not in name:
    #     name+=".png"
    #img = cv2.imread(folder + name, cv2.IMREAD_COLOR)
    #hist = arguments.hist
    img = cv2.imread("imagens/objetos1.png", cv2.IMREAD_GRAYSCALE)

    #B, G, R = cv2.split(img)
    B = img
    B = globalMethod(B)
    # G = globalMethod(G)
    # R = globalMethod(R)
    # final = cv2.merge((R, G, B))

    final = B
    plotAndSave(final)
    # if(opt == -1):
    #     print("Please insert an option as --option opt")
    # if(opt==1):
    #     globalMethod(img)
    
        