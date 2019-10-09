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

def binaryImage(img, nome="NaN", generate="y"):
    
    final = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (thresh, final) = cv2.threshold(final, 254, 255, cv2.THRESH_BINARY)
    if generate == "y":
        plotAndSave(final, nome, "binary_")

    return final

def grayscales(img, nome="NaN", generate="y"):
    
    final = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if generate=="y":
        plotAndSave(final, nome, "grayscale_")

    return final

def edge_detection(img, nome="NaN", generate="y"):
    
    final = binaryImage(img, nome, "n").astype('uint8')
    contours, _ = cv2.findContours(final, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    edges = cv2.drawContours(np.full(img.shape, 255), contours[1:], -1, (0, 0, 255))
    if generate=="y":
        plotAndSave(edges, nome, "edges_")

    return edges, contours

def contour_properties(img, nome="NaN", printHist = "n"):
    _, contours = edge_detection(img, nome, "n")
    edges = binaryImage(img.copy(), nome, "n")
    
    [peq, med, gra] = [0, 0, 0]
    if printHist=="n":
        print("Número de regiões: " + str(len(contours)-1) + "\n")
    
    i=0  
    for cont in (reversed(contours)):
        
        area = cv2.contourArea(cont)
        if(area<1500):
            peq += 1
        elif(area>=1500 and area<3000):
            med += 1
        else:
            gra += 1

        perimeter = cv2.arcLength(cont, True)
        
        hull = cv2.convexHull(cont)
        hull_area = cv2.contourArea(hull)
        solidity = area/hull_area

        (x, y), (MA, ma), angle = cv2.fitEllipse(cont)
        a = ma/2
        b = MA/2
        eccentricity = math.sqrt(pow(a, 2)-pow(b, 2))
        eccentricity = round(eccentricity/a, 2)

        M = cv2.moments(cont)
        if printHist=="n":
            print("Região {0:{1}d}: Área: {2:4.0f} Perímetro: {3:10.6f} Excentricidade: {4:10.6f} Solidez: {5:10.6f}"
                .format(i, 8, area, perimeter, eccentricity, solidity))
            
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        edges = printIn(edges, nome, cx, cy, i)
        i+=1
        if(i==len(contours)-1):
            break
    
    if(printHist=="n"):
        plotAndSave(edges, nome, "numbers_")
    else:
        print("número de regiões pequenas: " + str(peq) + "\nnúmero de regiões médias: " + str(med) + "\nnúmero de regiões grandes: " + str(gra))

        x_titles = ('Pequenas', 'Médias', 'Grandes')
        y_pos = np.arange(len(x_titles))
        sizes = [peq, med, gra]

        plt.bar(y_pos, sizes, align='center', alpha=1)
        plt.xticks(y_pos, x_titles)
        plt.ylabel('Número de objetos')
        plt.xlabel('Área')
        plt.title('Histograma de áreas de objetos')
        plt.savefig(f"hist_areas_{nome}.png")
        
def hist(img, nome="NaN"):
    contour_properties(img, nome, "y")

def printIn(img, nome, cx, cy, i):
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
        edge_detection(img, name)
    if(opt==4):
        contour_properties(img, name)
    if(opt==5):
        hist(img, name)
    
        