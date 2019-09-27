import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import warnings
#warnings.simplefilter("ignore")

def get_parser():
    parser = argparse.ArgumentParser(description='Global and local filters testing')
    parser.add_argument('--folder', default = '/home/marcelomartins/Documentos/8sem/imagens/proj2/imagens',
                        help='Folder where the image(s) are')#required = False é um argumento
    parser.add_argument('--option', type=int, choices = range(0,9), 
                        help = 'Choose filter you want to run')

    parser.add_argument('--mask', type=int, choices = range(0,100), default = 3,
                        help = 'Choose size of the mask')

    arguments = parser.parse_args()
    return arguments

def plotAndSave(method, black, size, img, final, mask=3):
    black = format((black*100/size), ".2f")
    plt.hist(img.ravel(),256,[0,256])
    cv2.imwrite(f"{method}_{mask}x{mask}.pgm", final)
    plt.savefig(f"{method}_histogram_{mask}x{mask}_{black}%black.png")

def initializeLocalMethod(mask, img):
    borda = math.floor(mask/2)
    black = 0
    size = (img.shape[0] - 2* borda) * (img.shape[1] - 2* borda)
    final = img.copy()

    return borda, black, size, final


def globalMethod(img):# 0 is height, 1 is weight but 0 is y and weight is x
    print("You chose global\n")

    final = img.copy() 
    black = 0
    size = img.shape[0] * img.shape[1]

    for y in range(0, img.shape[0]):
        for x in range(0, img.shape[1]):
            if(img[y, x] > 128):
                final[y,x] = 255
            else:
                final[y,x] = 0
                black+=1

    plotAndSave("global", black, size, img, final)
    
    
def bernsen(img, mask=3): #higher than T is black, otherwise is white
    print("You chose bernsen\n")

    borda, black, size, final = initializeLocalMethod(mask, img)

    for y in range(borda, img.shape[0] - borda):
        for x in range(borda, img.shape[1] -borda):
            temp = img[y-borda:y+borda+1, x-borda:x+borda+1]
            maxValue = np.amax(temp)
            minValue = np.amin(temp)
            result = (maxValue.astype('float') + minValue.astype('float'))/2
            if(img[y, x] > result):
                final[y,x] = 255
            else:
                final[y,x] = 0
                black+=1
    
    plotAndSave("bernsen", black, size, img, final, mask)

def niblack(img, mask=3): #mascara de 3x3
    print("You chose niblack\n")

    k=0.2
    borda, black, size, final = initializeLocalMethod(mask, img)

    for y in range(borda, img.shape[0] - borda):
        for x in range(borda, img.shape[1] -borda):
            temp = img[y-borda:y+borda+1, x-borda:x+borda+1]
            desvio = np.std(temp, dtype='float')
            media = np.mean(temp, dtype='float')
            result = media + k*desvio
            if(img[y, x] > result):
                final[y,x] = 0
            else:
                final[y,x] = 255
                black+=1
            
    plotAndSave("niblack", black, size, img, final, mask)

def sauvola(img, mask=3): #mascara de 3x3
    print("You chose sauvola\n")

    borda, black, size, final = initializeLocalMethod(mask, img)
    k = 0.5
    r = 128

    for y in range(borda, img.shape[0] - borda):
        for x in range(borda, img.shape[1] -borda):
            temp = img[y-borda:y+borda+1, x-borda:x+borda+1]
            desvio = np.std(temp, dtype='float')
            media = np.mean(temp, dtype='float')
            result = media * (1 + k * (desvio/r -1))
            if(img[y, x] > result):
                final[y, x] = 255
            else:
                final[y, x] = 0
                black+=1
            
    plotAndSave("sauvola", black, size, img, final, mask)

def sabale(img, mask=3): #mascara de 3x3
    print("You chose sabale\n")

    borda, black, size, final = initializeLocalMethod(mask, img)
    k = 0.25
    r = 0.5
    p = 2
    q = 10
    for y in range(borda, img.shape[0] - borda):
        for x in range(borda, img.shape[1] -borda):
            temp = img[y-borda:y+borda+1, x-borda:x+borda+1]
            desvio = np.std(temp, dtype='float')
            media = np.mean(temp, dtype='float')
            result = media * (1 + p*math.exp(-q* media) + k*(desvio/r -1))
            if(img[y, x] > result):
                final[y,x] = 255
            else:
                final[y,x] = 0
                black+=1
            
    plotAndSave("sabale", black, size, img, final, mask)

def contrast(img, mask=3): #mascara de 3x3
    print("You chose constrast\n")

    borda, black, size, final = initializeLocalMethod(mask, img)

    for y in range(borda, img.shape[0] - borda):
        for x in range(borda, img.shape[1] -borda):
            temp = img[y-borda:y+borda+1, x-borda:x+borda+1]
            maxValue = np.amax(temp).astype('float')
            minValue = np.amin(temp).astype('float')
            distToMax = abs(img[y, x] - maxValue)
            distToMin = abs(img[y, x] - minValue)
            if(distToMax > distToMin):
                final[y,x] = 0
                black+=1
            else:
                final[y,x] = 255
            
    plotAndSave("contrast", black, size, img, final, mask)

def mean(img, mask=3): #mascara de 3x3
    print("You chose mean\n")

    borda, black, size, final = initializeLocalMethod(mask, img)
    
    for y in range(borda, img.shape[0] - borda):
        for x in range(borda, img.shape[1] -borda):
            temp = img[y-borda:y+borda+1, x-borda:x+borda+1]
            media = np.mean(temp, dtype='float')
            if(img[y, x] > media):
                final[y,x] = 0
                black+=1
            else:
                final[y,x] = 255
            
    plotAndSave("mean", black, size, img, final, mask)

def mediana(img, mask=3): #mascara de 3x3
    print("You chose mediana\n")

    borda, black, size, final = initializeLocalMethod(mask, img)

    for y in range(borda, img.shape[0] - borda):
        for x in range(borda, img.shape[1] -borda):
            temp = img[y-borda:y+borda+1, x-borda:x+borda+1]
            median = np.median(temp)
            if(img[y, x] > median):
                final[y,x] = 0
                black+=1
            else:
                final[y,x] = 255
            
    plotAndSave("mediana", black, size, img, final, mask)

if __name__ == '__main__':
    
    arguments = get_parser()
    img = cv2.imread('imagens/fiducial.pgm', cv2.IMREAD_UNCHANGED)
    opt = int(arguments.option)
    mask = int(arguments.mask)
    
    # count = 0
    # opt = 1
    # mask = 3
    # while(True):
    #     if(opt==1):
    #         globalMethod(img)
    #     elif(opt==2):
    #         bernsen(img, mask)
    #     elif(opt==3):
    #         niblack(img, mask)
    #     elif(opt==4):
    #         sauvola(img, mask)
    #     elif(opt==5):
    #         sabale(img, mask)
    #     elif(opt==6):
    #         contrast(img, mask)
    #     elif(opt==7):
    #         mean(img, mask)
    #     elif(opt==8):
    #         mediana(img, mask)
    #     else:
    #         mask+=2
    #         opt = 1
    #         if(mask==9):
    #             break
    #     opt += 1

    if(opt==1):
        globalMethod(img)
    elif(opt==2):
        bernsen(img, mask)
    elif(opt==3):
        niblack(img, mask)
    elif(opt==4):
        sauvola(img, mask)
    elif(opt==5):
        sabale(img, mask)
    elif(opt==6):
        contrast(img, mask)
    elif(opt==7):
        mean(img, mask)
    elif(opt==8):
        mediana(img, mask)
