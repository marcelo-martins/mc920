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
                        help='Folder where the image(s) are')
    parser.add_argument('--option', type=int, choices = range(1,7), default = -1,
                        help = 'Choose option you want to run\n1 : binary\n2 : grayscale\n3 : detect_contours'
                            + '\n4 : generate area, perimeter, eccentricity and solitidy\n'
                            + '5 : generate area histogram\n6 : generate all\n')
    parser.add_argument('--image', default = "objetos3", help = 'Choose name of the PNG image you want to run')

    arguments = parser.parse_args()
    return arguments

def plotAndSave(img, nome = "NaN", func = "NaF"): #save images
    cv2.imwrite(f"{func}{nome}.png", img)

def binaryImage(img, nome="NaN", generate="y"): #generate binary image
    
    final = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (thresh, final) = cv2.threshold(final, 254, 255, cv2.THRESH_BINARY)
    if generate == "y":
        plotAndSave(final, nome, "binary_")

    return final

def grayscales(img, nome="NaN", generate="y"): #generate grayscale image
    
    final = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if generate=="y":
        plotAndSave(final, nome, "grayscale_")

    return final

def detect_contours(img, nome="NaN", generate="y"): #generate contours
    
    final = binaryImage(img, nome, "n").astype('uint8')
    contours, _ = cv2.findContours(final, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    edges = cv2.drawContours(np.full(img.shape, 255), contours[1:], -1, (0, 0, 255))
    if generate=="y":
        plotAndSave(edges, nome, "edges_")

    return edges, contours

def contour_properties(img, nome="NaN", printHist = "n"): #generate all the properties asked
    _, contours = detect_contours(img, nome, "n")
    edges = binaryImage(img.copy(), nome, "n")
    
    [peq, med, gra] = [0, 0, 0]
    if printHist=="n":
        print("Número de regiões: " + str(len(contours)-1) + "\n")

    i=0
    max_area = 0
    sizes = []
    for cont in (reversed(contours)):
        
        area = cv2.contourArea(cont) #area
        if(area<1500):
            peq += 1
        elif(area>=1500 and area<3000):
            med += 1
        else:
            gra += 1
        if(area>max_area):
            max_area = area
        sizes.append(area)

        perimeter = cv2.arcLength(cont, True) #perimeter
        
        hull = cv2.convexHull(cont)
        hull_area = cv2.contourArea(hull)
        solidity = area/hull_area #solidity

        (_, _), (MA, ma), _ = cv2.fitEllipse(cont) #eccentricity
        a = ma/2
        b = MA/2
        eccentricity = math.sqrt(pow(a, 2)-pow(b, 2))
        eccentricity = round(eccentricity/a, 2)

        M = cv2.moments(cont)
        if printHist=="n":
            print("Região {0:{1}d}: Área: {2:4.0f} Perímetro: {3:10.6f} Excentricidade: {4:10.6f} Solidez: {5:10.6f}"
                .format(i, 8, area, perimeter, eccentricity, solidity))
            
        cx = int(M['m10']/M['m00']) #centroid
        cy = int(M['m01']/M['m00'])
        edges = printIn(edges, nome, cx, cy, i)
        i+=1
        if(i==len(contours)-1):
            break
    
    if(printHist=="n"):
        plotAndSave(edges, nome, "numbers_")
    else:
        print("número de regiões pequenas: " + str(peq) + "\nnúmero de regiões médias: " + str(med) + "\nnúmero de regiões grandes: " + str(gra))

        plt.ylabel('Número de objetos') #generate histogram
        plt.xlabel('Área')
        plt.title('Histograma de áreas de objetos')
        plt.hist(sizes, bins=[0, 1500, 3000, max(3000, max_area)], edgecolor = [0,0,0])
        plt.savefig(f"hist_areas_{nome}.png")
        
def hist(img, nome="NaN"):
    contour_properties(img, nome, "y")

def printIn(img, nome, cx, cy, i): #insert numbers on image
    font = cv2.FONT_HERSHEY_SIMPLEX
    writeIn = (cx-6,cy+3)
    fontSize = 0.3
    fontColor = (255,255,255)

    cv2.putText(img, str(i), writeIn, font, fontSize, fontColor, 1)
    
    return img

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
        detect_contours(img, name)
    if(opt==4):
        contour_properties(img, name)
    if(opt==5):
        hist(img, name)
    if(opt==6):
        binaryImage(img, name)
        grayscales(img, name)
        detect_contours(img, name)
        contour_properties(img, name)
        hist(img, name)
