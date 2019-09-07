import cv2;
import numpy as np;
import math
import matplotlib.pyplot as plt
import os
import glob

floyd = np.array([[0, 0, 7/16],
           [3/16, 5/16, 1/16]]).reshape(2,3)

stevenson = np.array([[0 , 0 , 0 , 0 , 0 , 32/200 , 0],
            [12/200, 0 , 26/200 , 0 , 30/200 , 0 , 16/200],
            [0 , 12/200 , 0 , 26/200 , 0 , 12/200 , 0],
            [5/200 , 0 , 12/200 , 0 , 12/200 , 0 , 5/200]]).reshape(4,7)

burkes = np.array([[0 , 0 , 0 , 8/32 , 4/32],
         [2/32 , 4/32 , 8/32 , 4/32 , 2/32]]).reshape(2,5)

sierra = np.array([[0 , 0 , 0 , 5/32 , 3/32],
            [2/32 , 4/32 , 5/32 , 4/32 , 2/32],
            [0 , 2/32 , 3/32 , 2/32 , 0]]).reshape(3, 5)

stucki = np.array([[0 , 0 , 0, 8/42 , 4/42],
            [2/42 , 4/42 , 8/42 , 4/42 , 2/42],
            [1/42 , 2/42 , 4/42 , 2/42 , 1/42]]).reshape(3,5)

jarvis = np.array([[0 , 0 , 0, 7/48 , 5/48],
            [3/48 , 5/48 , 7/48 , 5/48 , 3/48],
            [1/48 , 3/48 , 5/48 , 3/48 , 1/48]]).reshape(3,5)


#Getting all files
def getImages(color, image_dir):
    data_path = os.path.join(image_dir, '*g')
    files = glob.glob(image_dir + "/*.png")
    data=[]
    names = []
    if (color=="colorful"):
        for fl in files:
            img = cv2.imread(fl, cv2.IMREAD_COLOR)
            data.append(img)
            names.append(os.path.splitext(os.path.basename(fl))[0])
    elif(color == "gray"):
        for fl in files:
            img = cv2.imread(fl, cv2.IMREAD_GRAYSCALE)
            data.append(img)
            names.append(os.path.splitext(os.path.basename(fl))[0])
    else:
        print("please specify color type")
    return data, names

def goThroughSnake(matrix, bordersize, filtro):
    startOfFilter = math.floor(filtro.shape[1]/2)

    (row,column) = matrix.shape
    print(row, column)
    matrixAux = matrix.copy()
    row-= bordersize
    column-=bordersize
    interval = [column, bordersize]
    one = 1
    #origem fica em cima
    for i in range(bordersize, row):#i row, j column
        temp = interval[0]
        interval[0] = interval[1]
        interval[1] = temp
        one = -one
        for j in range(interval[1], interval[0], one):
            if(matrix[i,j] < 128):
                matrixAux[i,j] = 0
            else:
                matrixAux[i,j] = 255
            err = float(matrix[i,j] - matrixAux[i,j])
            for g in range(0, filtro.shape[0]):
                for k in range(0, filtro.shape[1]):
                    matrix[i+g, j+((k-startOfFilter)*one)] += filtro[g,k] * err
    return matrixAux

def goThroughStraight(matrix, bordersize, filtro):
    startOfFilter = math.floor(filtro.shape[1]/2)

    (row,column) = matrix.shape
    print(row, column)
    matrixAux = matrix.copy()
    row-= bordersize
    column-=bordersize

    for i in range(bordersize, row):#i row, j column
        for j in range(bordersize, column):
            if(matrix[i,j] < 128):
                matrixAux[i,j] = 0
            else:
                matrixAux[i,j] = 255
            err = float(matrix[i,j] - matrixAux[i,j])
            for g in range(0, filtro.shape[0]):
                for k in range(0, filtro.shape[1]):
                    matrix[i+g, j+(k-startOfFilter)] += filtro[g,k] * err
    return matrixAux


def testStraightColor(title, matrixColor, filtro, filterName):
  
    matrixB, matrixG, matrixR = cv2.split(matrixColor)

    bordersize=5
    matrixR = cv2.copyMakeBorder(matrixR, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT, value=[255,255,255])
    matrixG = cv2.copyMakeBorder(matrixG, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT, value=[255,255,255])
    matrixB = cv2.copyMakeBorder(matrixB, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT, value=[255,255,255])

    #chama a funcao pras 3
    matrixR = goThroughStraight(matrixR, bordersize, filtro)
    matrixG = goThroughStraight(matrixG, bordersize, filtro)
    matrixB = goThroughStraight(matrixB, bordersize, filtro)

    matrixColor = cv2.merge((matrixB, matrixG, matrixR))
    final = matrixColor[bordersize:matrixColor.shape[1]-bordersize,bordersize:matrixColor.shape[0]-bordersize].copy()
    cv2.imwrite(title + 'ColorStraight' + filterName+'.png', cv2.bitwise_not(final))


def testSnakesColor(title, matrixColor, filtro, filterName):
   
    matrixB, matrixG, matrixR = cv2.split(matrixColor)

    bordersize=5
    matrixR = cv2.copyMakeBorder(matrixR, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT, value=[255,255,255])
    matrixG = cv2.copyMakeBorder(matrixG, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT, value=[255,255,255])
    matrixB = cv2.copyMakeBorder(matrixB, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT, value=[255,255,255])

    #chama a funcao pras 3
    matrixR = goThroughSnake(matrixR, bordersize, filtro)
    matrixG = goThroughSnake(matrixG, bordersize, filtro)
    matrixB = goThroughSnake(matrixB, bordersize, filtro)

    matrixColor = cv2.merge((matrixB, matrixG, matrixR))
    final = matrixColor[bordersize:matrixColor.shape[1]-bordersize,bordersize:matrixColor.shape[0]-bordersize].copy()
    cv2.imwrite(title + 'ColorSnake' + filterName + '.png', cv2.bitwise_not(final))


def testStraightGray(title, matrixGray, filtro, filterName):
   
    bordersize=5
    matrixGray = cv2.copyMakeBorder(matrixGray, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT, value=[0,0,0])
    matrixGray = goThroughStraight(matrixGray, bordersize, filtro)

    #cut border out
    matrixFinal = matrixGray[bordersize:matrixGray.shape[1]-bordersize,bordersize:matrixGray.shape[0]-bordersize].copy()
    cv2.imwrite(title + 'GrayStraight' + filterName + '.png', matrixFinal)

def testSnakesGray(title, matrixGray, filtro, filterName):

    bordersize=5
    matrixGray = cv2.copyMakeBorder(matrixGray, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT, value=[0,0,0])

    matrixGray = goThroughSnake(matrixGray, bordersize, filtro)

    #cut border out
    matrixFinal = matrixGray[bordersize:matrixGray.shape[1]-bordersize,bordersize:matrixGray.shape[0]-bordersize].copy()
    cv2.imwrite(title + 'GraySnake' + filterName + '.png', matrixFinal)


if __name__ == '__main__':

    allFilters = [floyd, stevenson, burkes, sierra, stucki, jarvis]
    allFiltersNames = ["floyd", "stevenson", "burkes", "sierra", "stucki", "jarvis"]

    #Color images straight
    data, names = getImages("colorful", "/home/marcelomartins/Documentos/8sem/imagens/proj1/images")
    for filtro in range(len(allFilters)):
        print(filtro)
        for i in range(len(data)):
            testStraightColor(names[i], data[i], allFilters[filtro], allFiltersNames[filtro])

    #Gray images snake
    data, names = getImages("gray", "/home/marcelomartins/Documentos/8sem/imagens/proj1/images")
    for filtro in range(len(allFilters)):
        for i in range(len(data)):
            testSnakesGray(names[i], data[i], allFilters[filtro], allFiltersNames[filtro])

    #Gray images straight
    data, names = getImages("gray", "/home/marcelomartins/Documentos/8sem/imagens/proj1/images")
    for filtro in range(len(allFilters)):
        for i in range(len(data)):
            testStraightGray(names[i], data[i], allFilters[filtro], allFiltersNames[filtro])