import cv2
import numpy as np
import sys
import argparse
import matplotlib.pyplot as plt
from argparse import RawTextHelpFormatter
from numpy.linalg import svd
import os
import warnings

warnings.simplefilter("ignore")

def get_parser():
    parser = argparse.ArgumentParser(description='PCA', formatter_class=RawTextHelpFormatter)
    parser.add_argument('--folder', default = 'images/',
                        help='Folder where the image(s) are')
    parser.add_argument('--k', default = 10, help = 'Components number')
    parser.add_argument('--out', default = 'generated', help='Out folder')
    parser.add_argument('--image', default = "baboon", help = 'Choose name of the PNG image you want to run')
    parser.add_argument('--all', default='n', help = 'Components number')
    arguments = parser.parse_args()
    return arguments

def plotAndSave(img, path, name = "NaN", k = "NaF"):#Save images
    cv2.imwrite(os.path.join(path, f"{name}_k{k}.png"), img, [cv2.IMWRITE_PNG_COMPRESSION, 9])

def get_compress_rate(out_folder, image_name, k_image, folder):
    compressed_size = os.path.getsize(out_folder + f"/{image_name}_k{k_image}.png")
    not_compressed_size = os.path.getsize(folder + f"/{image_name}.png")
    rate = compressed_size/not_compressed_size
    print(f"K = {k_image}")
    print("Compress rate = {0:1.4f}".format(rate))
    return round(rate,2)

def get_rmse(img, final):
    rmse = np.sqrt(np.mean(np.square(img.astype('float') - final.astype('float'))))
    print("Root mean square error(RSME) = {0:4.5f}\n".format(rmse))
    return round(rmse, 2)

def PCA(image_name, k_image, folder, out_folder):
    if '.png' not in image_name:
        image_name += '.png'

    img = cv2.imread(folder + image_name)
    img_original = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype('double')

    r_u, r_s, r_v = svd(img[:, :, 0], full_matrices=False)#svd returns, U, S and Vh. full_matrices tell the svd function to grab
                                                          #min(M, N) = k as one of the shapes of U and Vh
    g_u, g_s, g_v = svd(img[:, :, 1], full_matrices=False)
    b_u, b_s, b_v = svd(img[:, :, 2], full_matrices=False)

    k_real = min(img.shape[0], img.shape[1])
    if(k_real<k_image):
        print(f"Warning: your chosen K is higher than expected, only {k_real} components are going to be used")
        k_image = k_real

    #Merge components
    u_total = cv2.merge((r_u, g_u, b_u))
    s_total = cv2.merge((np.diag(r_s), np.diag(g_s), np.diag(b_s)))
    v_total = cv2.merge((r_v, g_v, b_v))

    g_total = np.zeros(shape=(img.shape[0], img.shape[1]))

    u_final = u_total[:, :k_image, :]
    s_final = s_total[:k_image, :k_image, :]
    v_final = v_total[:k_image, :, :]

    bus = np.matmul(u_final[:, :, 0], s_final[:, :, 0])
    gus = np.matmul(u_final[:, :, 1], s_final[:, :, 1])
    rus = np.matmul(u_final[:, :, 2], s_final[:, :, 2])

    #Multiply everything
    b_final = np.matmul(bus, v_final[:, :, 0])
    g_final = np.matmul(gus, v_final[:, :, 1])
    r_final = np.matmul(rus, v_final[:, :, 2])

    final = cv2.merge((r_final, g_final, b_final))
    image_name = image_name[:-4]
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
        print(f"Folder `{out_folder}` created to store your images")
    plotAndSave(final, out_folder, image_name, k_image)

    compress_data = get_compress_rate(out_folder, image_name, k_image, folder)
    rmse_data = get_rmse(img_original, final)

    return compress_data, rmse_data

def Main():
    #Reading image and text file
    arguments = get_parser()
    image_name = arguments.image
    k_image = int(arguments.k)
    folder = arguments.folder
    out_folder = arguments.out
    all_image = arguments.all

    if(all_image=='n'):
        PCA(image_name, k_image, folder, out_folder)
        return

    if '.png' not in image_name:
        image_name += '.png'
    img = cv2.imread(folder + image_name)
    
    sizes = [1, 5, 10, 20, 30, 40, 50]
    maximum = 100
    while(maximum<=max(img.shape[0], img.shape[1])):
        sizes.append(maximum)
        maximum += 100
    
    rmse_data = []
    compress_data = []
    for i in sizes:
        compress_ret, rmse_ret = PCA(image_name, i, folder, out_folder)
        compress_data.append(compress_ret)
        rmse_data.append((rmse_ret))

    image_name = image_name[:-4]
    
    #Generate compress rate graph
    fig, ax = plt.subplots()
    plt.title(f"Compress rate {image_name}")
    ax.plot(sizes, compress_data)
    ax.set(xlabel='k', ylabel='compress rate')
    plt.savefig(out_folder + f"/{image_name}_compress_graph")

    fig, ax = plt.subplots()
    plt.title(f"RMSE {image_name}")
    ax.plot(sizes, rmse_data)
    ax.set(xlabel='k', ylabel='rmse')
    plt.savefig(out_folder + f"/{image_name}_rmse_graph")

if __name__ == '__main__':
    Main()