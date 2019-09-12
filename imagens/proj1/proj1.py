import cv2;
import numpy as np;
import math
import matplotlib.pyplot as plt
import os
import glob
import sys

#Filters

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


#Getting all files at once
def getImages(color, image_dir):
    data_path = os.path.join(image_dir, '*g')
    files = glob.glob(image_dir + "/*.png")
    data=[]
    names = []
    if (color=="colorful"): #Getting all colorful as they already are
        for fl in files:
            img = cv2.imread(fl, cv2.IMREAD_COLOR)
            data.append(img)
            names.append(os.path.splitext(os.path.basename(fl))[0])
    elif(color == "gray"): #Converting then to grayScale
        for fl in files:
            img = cv2.imread(fl, cv2.IMREAD_GRAYSCALE)
            data.append(img)
            names.append(os.path.splitext(os.path.basename(fl))[0])
    else:
        print("please specify color type")
    return data, names

def goThroughSnake(matrix, bordersize, filtro): #Go through the matrix as a snake
    
    startOfFilter = math.floor(filtro.shape[1]/2) #Get f(x,y)
    (row,column) = matrix.shape
    matrixAux = matrix.copy()
    row-= bordersize #Subtract the bordersize to run only on the original matrix
    column-=bordersize
    interval = [column, bordersize]
    one = 1 #Variable to change orientation as rows go by
    
    for i in range(bordersize, row):#i row, j column
        temp = interval[0] #Change interval. For even rows we go (bordersize, column), and for odd, (column, bordersize)
        interval[0] = interval[1]
        interval[1] = temp
        one = -one
        for j in range(interval[1], interval[0], one):
            if(matrix[i,j] < 128):
                matrixAux[i,j] = 0
            else:
                matrixAux[i,j] = 255
            err = float(matrix[i,j] - matrixAux[i,j])
            for g in range(0, filtro.shape[0]): #For loop that applies the filters on f(x,y)'s neighbours
                for k in range(0, filtro.shape[1]):
                    matrix[i+g, j+((k-startOfFilter)*one)] += filtro[g,k] * err
    return matrixAux

def goThroughStraight(matrix, bordersize, filtro): #This is pretty much the same function as Snake. 
#The only real difference here is that the variable "one" does not exist, as we don't need to change orientation

    startOfFilter = math.floor(filtro.shape[1]/2)
    (row,column) = matrix.shape
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
    #On this function, we test colorful images calling the straight method
    #In order to do that, we split the original matrix in its R, G, B components and call the straight method for each one
    #Before calling, we first apply the border to all of them
    #Before saving, we merge them and cut out the border

    matrixB, matrixG, matrixR = cv2.split(matrixColor)

    bordersize=5
    matrixR = cv2.copyMakeBorder(matrixR, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT, value=[255,255,255])
    matrixG = cv2.copyMakeBorder(matrixG, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT, value=[255,255,255])
    matrixB = cv2.copyMakeBorder(matrixB, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT, value=[255,255,255])

    #Calling function for each one
    matrixR = goThroughStraight(matrixR, bordersize, filtro)
    matrixG = goThroughStraight(matrixG, bordersize, filtro)
    matrixB = goThroughStraight(matrixB, bordersize, filtro)

    matrixColor = cv2.merge((matrixB, matrixG, matrixR))
    #Cutting border out
    final = matrixColor[bordersize:matrixColor.shape[1]-bordersize,bordersize:matrixColor.shape[0]-bordersize].copy()
    cv2.imwrite(title + 'ColorStraight' + filterName+'.png', cv2.bitwise_not(final))

def testSnakesColor(title, matrixColor, filtro, filterName):
    #Same thing as the Straight one, the only difference is the function that is called
   
    matrixB, matrixG, matrixR = cv2.split(matrixColor)

    bordersize=5
    matrixR = cv2.copyMakeBorder(matrixR, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT, value=[255,255,255])
    matrixG = cv2.copyMakeBorder(matrixG, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT, value=[255,255,255])
    matrixB = cv2.copyMakeBorder(matrixB, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT, value=[255,255,255])

    #Calling function for each one
    matrixR = goThroughSnake(matrixR, bordersize, filtro)
    matrixG = goThroughSnake(matrixG, bordersize, filtro)
    matrixB = goThroughSnake(matrixB, bordersize, filtro)

    matrixColor = cv2.merge((matrixB, matrixG, matrixR))
    #Cutting border out
    final = matrixColor[bordersize:matrixColor.shape[1]-bordersize,bordersize:matrixColor.shape[0]-bordersize].copy()
    cv2.imwrite(title + 'ColorSnake' + filterName + '.png', cv2.bitwise_not(final))

def testStraightGray(title, matrixGray, filtro, filterName):
    #Same thing as the colorful one, but here we only have one matrix
   
    bordersize=5
    matrixGray = cv2.copyMakeBorder(matrixGray, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT, value=[0,0,0])
    matrixGray = goThroughStraight(matrixGray, bordersize, filtro)

    #Cutting border out
    matrixFinal = matrixGray[bordersize:matrixGray.shape[1]-bordersize,bordersize:matrixGray.shape[0]-bordersize].copy()
    cv2.imwrite(title + 'GrayStraight' + filterName + '.png', cv2.bitwise_not(matrixFinal))

def testSnakesGray(title, matrixGray, filtro, filterName):

    bordersize=5
    matrixGray = cv2.copyMakeBorder(matrixGray, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT, value=[0,0,0])

    matrixGray = goThroughSnake(matrixGray, bordersize, filtro)

    #Cutting border out
    matrixFinal = matrixGray[bordersize:matrixGray.shape[1]-bordersize,bordersize:matrixGray.shape[0]-bordersize].copy()
    cv2.imwrite(title + 'GraySnake' + filterName + '.png', cv2.bitwise_not(matrixFinal))


#These functions iterate through all images and filters
def runSnakesColor(data, names, allFilters, allFiltersNames):
    for filtro in range(len(allFilters)):
        for i in range(len(data)):
            print(f"\nGenerating {names[i]} + {allFiltersNames[filtro]} + snake colorful")
            testSnakesColor(names[i], data[i], allFilters[filtro], allFiltersNames[filtro])


def runStraightColor(data, names, allFilters, allFiltersNames):
    for filtro in range(len(allFilters)):
        for i in range(len(data)):
            print(f"\nGenerating {names[i]} + {allFiltersNames[filtro]} + straight colorful")
            testStraightColor(names[i], data[i], allFilters[filtro], allFiltersNames[filtro])

def runSnakesGray(data, names, allFilters, allFiltersNames):
    for filtro in range(len(allFilters)):
        for i in range(len(data)):
            print(f"\nGenerating {names[i]} + {allFiltersNames[filtro]} + snake gray")
            testSnakesGray(names[i], data[i], allFilters[filtro], allFiltersNames[filtro])


def runStraightGray(data, names, allFilters, allFiltersNames):
    for filtro in range(len(allFilters)):
        for i in range(len(data)):
            print(f"\nGenerating {names[i]} + {allFiltersNames[filtro]} + straight gray")
            testStraightGray(names[i], data[i], allFilters[filtro], allFiltersNames[filtro])



if __name__ == '__main__':

    allFilters = [floyd, stevenson, burkes, sierra, stucki, jarvis]
    allFiltersNames = ["floyd", "stevenson", "burkes", "sierra", "stucki", "jarvis"]

    option = int(sys.argv[2])
    image_dir = sys.argv[1]

    suma1 = 0
    suma2 = 0
    suma3 = 0
    count = 0
    for i in range(1, sierra.shape[0]):
        for j in range(0, sierra.shape[1]):
            count +=1
            suma1 += sierra[i,j]/jarvis[i,j]
            suma2 += sierra[i,j]/stucki[i,j]
            suma3 += stucki[i,j]/jarvis[i,j]


    print(f"{suma1/count} {suma2/count} {suma3/count}")
        

    if(option==1): #Color Snake
        print("\nGenerating colorful snake\n-------------------------------------")
        data, names = getImages("colorful", image_dir)
        runSnakesColor(data, names, allFilters, allFiltersNames)
    elif(option==2): #Color Straight
        print("\nGenerating colorful straight\n-------------------------------------")
        data, names = getImages("colorful", image_dir)
        runStraightColor(data, names, allFilters, allFiltersNames)
    elif(option==3): #Gray Snake
        print("\nGenerating gray snake\n-------------------------------------")
        data, names = getImages("gray", image_dir)
        runSnakesGray(data, names, allFilters, allFiltersNames)
    elif(option==4): #Gray Straight
        print("\nGenerating gray straight\n-------------------------------------")
        data, names = getImages("gray", image_dir)
        runStraightGray(data, names, allFilters, allFiltersNames)
    elif(option==5): #All of them
        print("\nGenerating everything\n-------------------------------------")
        data, names = getImages("colorful", image_dir)
        runSnakesColor(data, names, allFilters, allFiltersNames)
        data, names = getImages("colorful", image_dir)
        runStraightColor(data, names, allFilters, allFiltersNames)
        data, names = getImages("gray", image_dir)
        runSnakesGray(data, names, allFilters, allFiltersNames)
        data, names = getImages("gray", image_dir)
        runStraightGray(data, names, allFilters, allFiltersNames)
    else:
        print("Invalid argument. Try again")