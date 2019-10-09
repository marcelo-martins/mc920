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

    return final

def grayscales(img, nome="NaN", generate="y"):
    
    final = img.copy() 
    final = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)
    
    if generate=="y":
        plotAndSave(final, nome, "grayscale_")

    return final

def edge_detection(img, nome="NaN", generate="y"):
    final = img.copy()
    v = np.median(final)                                         #sepa eh melhor tirar isso
    lower = int(max(0, (1.0 - 0.33) * v))
    upper = int(min(255, (1.0 + 0.33) * v))
    final = grayscales(final, nome, "n")
    edges = cv2.Canny(final, lower, upper)

    (thresh, edges) = cv2.threshold(edges, 254, 255, cv2.THRESH_BINARY)
    #edges = cv2.Canny(final,100,200)
    if generate=="y":
        plotAndSave(cv2.bitwise_not(edges), nome, "edges_")

    return edges

def contour_properties(img, nome="NaN", printHist = "n"):
    edges = edge_detection(img, nome, "n")
    contours, h = cv2.findContours(edges, 0, 3) #0,3
    
    [peq, med, gra] = [0, 0, 0]
    if printHist=="n":
        print("Número de regiões: " + str(len(contours)) + "\n")
    
    i=0
    for cont in contours:
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
        solidity = float(area)/hull_area

        (x, y), (MA, ma), angle = cv2.fitEllipse(cont)
        a = ma/2
        b = MA/2
        eccentricity = math.sqrt(pow(a, 2)-pow(b, 2))
        eccentricity = round(eccentricity/a, 2)

        if printHist == "n":
            print("Região " + str(i) + ": Área: " + str(format(math.ceil(area), "4.0f")) 
                + " Perímetro:  " + str(format(perimeter, "4.6f")) 
                + " Excentricidade: " + str(format(eccentricity, "4.6f")) 
                + " Solidez: " + str(format(solidity, "3.6f")))
        
        M = cv2.moments(cont)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        edges = printIn(edges, nome, cx, cy, i)
        i+=1
    
    if(printHist=="n"):
        plotAndSave(cv2.bitwise_not(edges), nome, "numbers_")
    else:
        print("número de regiões pequenas: " + str(peq) + "\nnúmero de regiões médias: " + str(med) + "\nnúmero de regiões grandes: " + str(gra))

        objects = ('Pequenas', 'Médias', 'Grandes')
        y_pos = np.arange(len(objects))
        performance = [peq, med, gra]

        plt.bar(y_pos, performance, align='center', alpha=1)
        plt.xticks(y_pos, objects)
        plt.ylabel('Número de objetos')
        plt.xlabel('Área')
        plt.title('Histograma de áreas de objetos')
        plt.savefig(f"hist_areas_{nome}.png")
        
def hist(img, nome="NaN"):
    contour_properties(img, nome, "y")

def printIn(img, nome, cx, cy, i):
    font = cv2.FONT_HERSHEY_SIMPLEX
    writeIn = (cx,cy)
    fontSize = 0.5
    fontColor = (255)

    cv2.putText(img, str(i), writeIn, font, fontSize, fontColor, 2)
    
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
    
        