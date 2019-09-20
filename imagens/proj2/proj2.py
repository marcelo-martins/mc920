import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math

def get_parser():
    parser = argparse.ArgumentParser(description='Global and local filters testing')
    parser.add_argument('--folder', default = '/home/marcelomartins/Documentos/8sem/imagens/proj2/imagens',
                        help='Folder where the image(s) are')#required = False é um argumento
    parser.add_argument('--option', type=int, choices = range(0,9), 
                        help = 'Choose filter you want to run')

    parser.add_argument('--mask', type=int, choices = range(0,100), default = 3,
                        help = 'Choose filter you want to run')

    arguments = parser.parse_args()
    return arguments


def globalMethod(img):# 0 is height, 1 is weight but 0 is y and weight is x

    final = img.copy()    
    for y in range(0, img.shape[0]):
        for x in range(0, img.shape[1]):
            #final = np.where(img < 128, 0, 255)
            if(img[y, x] > 128):
                final[y,x] = 0
            else:
                final[y,x] = 255
    cv2.imshow("nome", final)
    cv2.waitKey(0)


    #final = np.where(final < 128, 0, 255)
    
def bernsen(img, mask=3): #higher than T is black, otherwise is white

    borda = math.floor(mask/2)

    final = img.copy()
    for y in range(borda, img.shape[0] - borda):
        for x in range(borda, img.shape[1] -borda):
            temp = final[y-borda:y+borda+1, x-borda:x+borda+1]
            maxValue = np.amax(temp)
            minValue = np.amin(temp)

            result = (maxValue + minValue)/2
            if(img[y, x] > result):
                final[y,x] = 0
            else:
                final[y,x] = 255
    

    cv2.imshow("nome", final)
    cv2.waitKey(0)

def niblack(img, mask=3): #mascara de 3x3

    arrayK = [-0.3, -0.2, -0.1, 0.1, 0.2, 0.3]
    for i in range(0, len(arrayK)):
        k = arrayK[i] 
        borda = math.floor(mask/2)

        final = img.copy()
        for y in range(borda, img.shape[0] - borda):
            for x in range(borda, img.shape[1] -borda):
                temp = final[y-borda:y+borda+1, x-borda:x+borda+1]
                desvio = np.std(temp)
                media = temp.mean()
                
                result = media + k*desvio
                if(img[y, x] > result):
                    final[y,x] = 0
                else:
                    final[y,x] = 255
                
        cv2.imwrite(f"niblackK={k}.pgm", final)   

def sauvola(img, mask=3): #mascara de 3x3

    borda = math.floor(mask/2)

    final = img.copy()
    for y in range(borda, img.shape[0] - borda):
        for x in range(borda, img.shape[1] -borda):
            temp = final[y-borda:y+borda+1, x-borda:x+borda+1]
            desvio = np.std(temp)
            media = temp.mean()
            
            result = media * (1 + 0.5 * (desvio/128 -1))
            if(img[y, x] > result):
                final[y,x] = 0
            else:
                final[y,x] = 255
            
    cv2.imwrite("sauvola.pgm", final)   

def sabale(img, mask=3): #mascara de 3x3

    borda = math.floor(mask/2)

    final = img.copy()
    for y in range(borda, img.shape[0] - borda):
        for x in range(borda, img.shape[1] -borda):
            temp = final[y-borda:y+borda+1, x-borda:x+borda+1]
            desvio = np.std(temp)
            media = temp.mean()
            
            result = media * (1 + 2*math.exp(-10* media) + 0.25*(desvio/0.5 -1))
            if(img[y, x] > result):
                final[y,x] = 0
            else:
                final[y,x] = 255
            
    cv2.imwrite("sabale.pgm", final) 

def contrast(img, mask=3): #mascara de 3x3

    borda = math.floor(mask/2)

    final = img.copy()
    for y in range(borda, img.shape[0] - borda):
        for x in range(borda, img.shape[1] -borda):
            temp = final[y-borda:y+borda+1, x-borda:x+borda+1]
            
            maxValue = np.amax(temp)
            minValue = np.amin(temp)

            distToMax = abs(img[y, x] - maxValue)
            distToMin = abs(img[y, x] - minValue)

            if(distToMax > distToMin): #closer to minimum = black
                final[y,x] = 0
            else:
                final[y,x] = 255
            
    cv2.imwrite("constast.pgm", final)   

def mean(img, mask=3): #mascara de 3x3

    borda = math.floor(mask/2)

    final = img.copy()
    for y in range(borda, img.shape[0] - borda):
        for x in range(borda, img.shape[1] -borda):
            temp = final[y-borda:y+borda+1, x-borda:x+borda+1]
            media = temp.mean()
            
            if(img[y, x] > media):
                final[y,x] = 0
            else:
                final[y,x] = 255
            
    cv2.imwrite("media.pgm", final)

def mediana(img, mask=3): #mascara de 3x3

    print("sussa")

    borda = math.floor(mask/2)
    print(f"borda = {borda}")

    final = img.copy()
    for y in range(borda, img.shape[0] - borda):
        for x in range(borda, img.shape[1] -borda):
            temp = final[y-borda:y+borda+1, x-borda:x+borda+1]
            if(temp.shape[0]==0 or temp.shape[1]==0):
                print("saco")
            median = np.median(temp)
            
            if(img[y, x] > median):
                final[y,x] = 0
            else:
                final[y,x] = 255
            
    cv2.imwrite("mediana.pgm", final)  

if __name__ == '__main__':
    arguments = get_parser()
    print(arguments.option)
    img = cv2.imread('imagens/baboon.pgm', cv2.IMREAD_UNCHANGED)
    print(img.dtype)

    opt = int(arguments.option)

    mask = int(arguments.mask)
    print(mask)
    
    if(opt==1):
        globalMethod(img)
    elif(opt==2):
        bernsen(img)
    elif(opt==3):
        niblack(img)
    elif(opt==4):
        sauvola(img)
    elif(opt==5):
        sabale(img)
    elif(opt==6):
        contrast(img)
    elif(opt==7):
        mean(img)
    elif(opt==8):
        mediana(img, mask)