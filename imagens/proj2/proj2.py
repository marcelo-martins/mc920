import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import warnings
import os
import glob
import sys
from argparse import RawTextHelpFormatter

warnings.simplefilter("ignore")

def get_parser():
    parser = argparse.ArgumentParser(description='Global and local filters testing', formatter_class=RawTextHelpFormatter)
    parser.add_argument('--folder', default = 'imagens/',
                        help='Folder where the image(s) are')#required = False é um argumento
    parser.add_argument('--option', type=int, choices = range(1,10), default = -1,
                        help = 'Choose filter you want to run\n1 : global\n2 : bernsen\n3 : niblack\n4 : sauvola\n'
                            + '5 : sabale\n6 : contrast\n7 : mean\n8 : median\n')
    parser.add_argument('--mask', type=int, choices = range(0,100), default = 3, metavar = 'range 1-100',
                        help = 'Choose size of the mask')
    parser.add_argument('--image', default = "baboon", help = 'Choose name of the PGM image you want to run')
    parser.add_argument('--hist', default = 'y', choices = ["y", "n", "Y", "N"], help = 'Choose if you want to generate histogram y/n')

    arguments = parser.parse_args()
    return arguments

def plotAndSave(method, black, size, img, final, mask=3, hist = "y"):
    black = format((black*100/size), ".2f")
    if(hist != "y"): #-----------------------------------------------------------------------------------------------
        cv2.imwrite(f"{method}_{mask}x{mask}.png", final) #change format to png
    if(hist.lower() == 'y'):
        plt.hist(img.ravel(),256,[0,256])
        plt.savefig(f"{method}_histogram_{mask}x{mask}_{black}%black.png")
    else:
        print(f"Percentage of black pixels = {black}%\n")

def initializeLocalMethod(mask, img):
    borda = math.floor(mask/2)
    black = 0
    size = img.shape[0] * img.shape[1] 
    final = img.copy()
    height = img.shape[0]
    width = img.shape[1]

    return borda, black, size, final, height, width

def globalMethod(img, nome="nome", hist="y"):# 0 is height, 1 is weight but 0 is y and weight is x
    print("You chose global\n")

    final = img.copy() 
    black = 0
    size = img.shape[0] * img.shape[1]

    for y in range(0, img.shape[0]):
        for x in range(0, img.shape[1]):
            if(img[y, x] >= 128):
                final[y,x] = 255
            else:
                final[y,x] = 0
                black+=1

    black = format((black*100/size), ".2f")
    cv2.imwrite(f"{name}_global.png", final)
    if(hist.lower() == 'y'):
        plt.hist(img.ravel(),256,[0,256])
        plt.savefig(f"{nome}_global_histogram_{black}%black.png")
    else:
        print(f"Percentage of black pixels = {black}%\n")
       
def bernsen(img, mask=3, nome="nome", hist="y"): #higher than T is black, otherwise is white
    print("You chose bernsen\n")

    borda, black, size, final, height, width = initializeLocalMethod(mask, img)

    for y in range(height):
        for x in range(width):
            temp = img[max(0, y-borda) : min(y+borda+1, height), max(0, x-borda) : min(x+borda+1, width)]
            maxValue = np.max(temp)
            minValue = np.min(temp)
            result = (maxValue.astype('float') + minValue.astype('float'))/2
            if(img[y, x] >= result):
                final[y,x] = 255
            else:
                final[y,x] = 0
                black+=1
    
    plotAndSave(f"{nome}_bernsen", black, size, img, final, mask, hist)

def niblack(img, mask=3, nome="nome", hist="y"):
    print("You chose niblack\n")

    k=0.2
    borda, black, size, final, height, width = initializeLocalMethod(mask, img)

    for y in range(height):
        for x in range(width):
            temp = img[max(0, y-borda) : min(y+borda+1, height), max(0, x-borda) : min(x+borda+1, width)]
            desvio = np.std(temp, dtype='float')
            media = np.mean(temp, dtype='float')
            result = media + k*desvio
            if(img[y, x] >= result):
                final[y,x] = 255
            else:
                final[y,x] = 0
                black+=1
            
    plotAndSave(f"{nome}_niblack_k{k}", black, size, img, final, mask, hist)

def sauvola(img, mask=3, nome="nome", hist="y"):
    print("You chose sauvola\n")

    borda, black, size, final, height, width = initializeLocalMethod(mask, img)
    k = 0.5
    r = 128

    for y in range(height):
        for x in range(width):
            temp = img[max(0, y-borda) : min(y+borda+1, height), max(0, x-borda) : min(x+borda+1, width)]
            desvio = np.std(temp, dtype='float')
            media = np.mean(temp, dtype='float')
            result = media * (1 + k * ((desvio/r) -1))
            if(img[y, x] >= result):
                final[y, x] = 255
            else:
                final[y, x] = 0
                black+=1
            
    plotAndSave(f"{nome}_sauvola", black, size, img, final, mask, hist)

def sabale(img, mask=3, nome="nome", hist="y"):
    print("You chose sabale\n")

    borda, black, size, final, height, width = initializeLocalMethod(mask, img)
    k = 0.25
    r = 0.5
    p = 2
    q = 10
    for y in range(height):
        for x in range(width):
            temp = img[max(0, y-borda) : min(y+borda+1, height), max(0, x-borda) : min(x+borda+1, width)]
            desvio = np.std(temp, dtype='float')
            media = np.mean(temp, dtype='float')
            result = media * (1 + p*np.exp(-q* media) + k*((desvio/r) -1))
            if(img[y, x] >= result):
                final[y,x] = 255
            else:
                final[y,x] = 0
                black+=1
            
    plotAndSave(f"{nome}_sabale", black, size, img, final, mask, hist)

def contrast(img, mask=3, nome="nome", hist="y"):
    print("You chose constrast\n")

    borda, black, size, final, height, width = initializeLocalMethod(mask, img)

    for y in range(height):
        for x in range(width):
            temp = img[max(0, y-borda) : min(y+borda+1, height), max(0, x-borda) : min(x+borda+1, width)]
            maxValue = np.max(temp).astype('float')
            minValue = np.min(temp).astype('float')
            distToMax = maxValue - img[y, x]
            distToMin = img[y, x] - minValue
            if(distToMax >= distToMin):
                final[y,x] = 255
            else:
                final[y,x] = 0
                black +=1
            
    plotAndSave(f"{nome}_contrast", black, size, img, final, mask, hist)

def mean(img, mask=3, nome="nome", hist="y"):
    print("You chose mean\n")

    borda, black, size, final, height, width = initializeLocalMethod(mask, img)
    
    for y in range(height):
        for x in range(width):
            temp = img[max(0, y-borda) : min(y+borda+1, height), max(0, x-borda) : min(x+borda+1, width)]
            media = np.mean(temp, dtype='float')
            if(img[y, x] >= media):
                final[y,x] = 255
            else:
                final[y,x] = 0
                black+=1
            
    plotAndSave(f"{nome}_mean", black, size, img, final, mask, hist)

def mediana(img, mask=3, nome="nome", hist="y"):
    print("You chose mediana\n")

    borda, black, size, final, height, width = initializeLocalMethod(mask, img)

    for y in range(height):
        for x in range(width):
            temp = img[max(0, y-borda) : min(y+borda+1, height), max(0, x-borda) : min(x+borda+1, width)]
            median = np.median(temp)
            if(img[y, x] >= median):
                final[y,x] = 255
            else:
                final[y,x] = 0
                black+=1
            
    plotAndSave(f"{nome}_mediana", black, size, img, final, mask, hist)

def getImages(image_dir):
    data_path = os.path.join(image_dir, '*g')
    files = glob.glob(image_dir + "/*.pgm")
    data=[]
    names = []
    for fl in files:
        img = cv2.imread(fl, cv2.IMREAD_UNCHANGED)
        data.append(img)
        names.append(os.path.splitext(os.path.basename(fl))[0])
    return data, names

if __name__ == '__main__':
    
    arguments = get_parser()
    opt = int(arguments.option)
    mask = int(arguments.mask)
    folder = arguments.folder
    name = arguments.image
    img = cv2.imread(folder + name + '.pgm', cv2.IMREAD_UNCHANGED)
    hist = arguments.hist

    if(opt == -1):
        print("Please insert an option as --option opt")
    elif(opt==1):
        globalMethod(img, name, hist)
    elif(opt==2):
        bernsen(img, mask, name, hist)
    elif(opt==3):
        niblack(img, mask, name, hist)
    elif(opt==4):
        sauvola(img, mask, name, hist)
    elif(opt==5):
        sabale(img, mask, name, hist)
    elif(opt==6):
        contrast(img, mask, name, hist)
    elif(opt==7):
        mean(img, mask, name, hist)
    elif(opt==8):
        mediana(img, mask, name, hist)
    elif(opt==9):
        print(f"\nExecuting everything for mask = {mask}\n------------------------------------------------------------------")
        globalMethod(img, name, hist)
        bernsen(img, mask, name, hist)
        niblack(img, mask, name, hist)
        sauvola(img, mask, name, hist)
        sabale(img, mask, name, hist)
        contrast(img, mask, name, hist)
        mean(img, mask, name, hist)
        mediana(img, mask, name, hist)
        