from SOM_Kohonen import SOM_Kohonen
import array
import os
import struct
import random
import math
import numpy as np
from PIL import Image
import cv2

def color(picture):
    max_width,max_height = picture.size
    data = []
    for wid in range(max_width):
        for hei in range(max_height):
            RGB = picture.getpixel((wid,hei))
            data.append([RGB[0],RGB[1],RGB[2]])
    return data

def div(picture, widthX, heightX, frame_size):
    tab = []
    for w in range(widthX,widthX+frame_size):
        for h in range(heightX,heightX+frame_size):
            tab.append(picture[w,h])
    return tab

def grey(picture, frame_size):
    max_width,max_height = picture.shape
    width = math.floor(max_width/frame_size)
    height = math.floor(max_height/frame_size)
    data = []
    for wid in range(width):
        for hei in range(height):
            data.append(div(picture, frame_size*wid, frame_size*hei, frame_size))
    return data 

def inputMaxMin(Input_amount):
    tab = []
    for i in range(Input_amount):
        tab.append([0,255])
    return tab

def make_dictionary(fileContent, Neuron_amount, Input_amount):
    dictionary = []
    for i in range(Neuron_amount):
        dictionary.append(tuple(fileContent[i*Input_amount:i*Input_amount+Input_amount]))
    return tuple(dictionary)

def make_data(fileContent, dictionary_end):
    data = []
    for i in fileContent[dictionary_end:]:
        data.append(i)
    return data

def unPack_color(picture, Neuron_amount, Input_amount, path, name):
    name2 = os.path.join(path, str(name) + ".txt")
    newFile2 = open(name2, "rb")
    fileContent = newFile2.read()
    dictionary = make_dictionary(fileContent, Neuron_amount, Input_amount)
    dictionary_end = Neuron_amount*Input_amount
    max_width,max_height = picture.size
    data = make_data(fileContent,dictionary_end)
    for wid in range(max_width):
        for hei in range(max_height):
            x = wid*max_height+hei
            pixel = data[x]
            picture.putpixel((wid, hei), dictionary[pixel])
    picture.show()
    picture.save(path + name + '.png')
    picture.close()

def set_grey(picture, wid, hei, dictionary, pixel, frame_size):
    for w, i in zip(range(wid, wid+frame_size), range(0,frame_size)):
        for h, j in zip(range(hei, hei+frame_size), range(0,frame_size)):
            x = i*frame_size+j
            picture[w,h]=dictionary[pixel][x]

def unpack_grey(picture, Neuron_amount, Input_amount, path, name, frame_size):
    name2 = os.path.join(path, str(name) + ".txt")
    newFile2 = open(name2, "rb")
    fileContent = newFile2.read()
    dictionary = make_dictionary(fileContent, Neuron_amount, Input_amount)
    dictionary_end = Neuron_amount*Input_amount
    max_width,max_height = picture.shape
    width = math.floor(max_width/frame_size)
    height = math.floor(max_height/frame_size)
    data = make_data(fileContent,dictionary_end)
    for wid in range(width):
        for hei in range(height):
            x = data[wid*height+hei]
            w = wid*frame_size
            h = hei*frame_size
            set_grey(picture, w, h, dictionary, x, frame_size)
    # cv2.imshow('ImageWindow', picture)
    # cv2.waitKey()
    name3 =os.path.join(path, str(name) + '.png')
    cv2.imwrite(name3, picture)

############################################### Control Panel ###################################################

######################### Image ########################
frame_size = 3
Neuron_amount = 20

if not os.path.exists('out'):
    os.makedirs('out')
Path = os.path.join(os.getcwd(), 'out')

picture = Image.open("2.jpg")
choice = 'c'
Name = 'name'

#################################################################################################################
if choice == 'c':
    Data = color(picture)
if choice == 'g':
    Data = grey(picture, frame_size)

########################## Data ########################
Input_amount = len(Data[0])
Input_min_max_value = inputMaxMin(Input_amount)

###################### SOM Kohonen #####################
Lr_max = 0.5
Lr_min = 0.01
Neighbourhood_max = 4
Neighbourhood_min = 0.01

#################################################################################################################
mapa = SOM_Kohonen(Input_amount, Neuron_amount, Lr_min, Lr_max, Neighbourhood_min, Neighbourhood_max, Input_min_max_value)
mapa.learn(Data)

################################################################################################################
if choice == 'c':
    Data2 = color(picture)
if choice == 'g':
    Data2 = grey(picture, frame_size)
mapa.save(Data2,Path,Name)
if choice == 'c':
    unPack_color(picture, Neuron_amount, Input_amount, Path, Name)
if choice == 'g':
    unpack_grey(picture, Neuron_amount, Input_amount, Path, Name, frame_size)
