import numpy as np
import math 
import array
import os
import struct
from Neuron import Neuron
from PIL import Image

class SOM_Kohonen(object):
    def __init__(self, input_amount, neuron_amount, lr_min, lr_max, neighbourhood_min, neighbourhood_max, input_min_max_value, potential_min=0.75):
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.lr = 0
        self.neighbourhood_min = neighbourhood_min
        self.neighbourhood_max = neighbourhood_max
        self.neighbourhood = 0
        self.potential_min = potential_min
        self.activity_tab = [1 for i in range(neuron_amount)]
        self.neurons = [Neuron(input_amount, input_min_max_value, i) for i in range(neuron_amount)]

    def find_winner(self, one_input):
        distance = []
        for neuron, activity in zip(self.neurons, self.activity_tab):
            if activity == 1:
                distance.append(neuron.distance_measure(one_input))
            else:
                distance.append(999999)
        return np.argmin(distance, axis=0)

    def lr_neighbourhood_update(self, step_number, steps_amount):
        self.lr = self.lr_max*pow(self.lr_min/self.lr_max, (step_number)/(steps_amount))
        self.neighbourhood = self.neighbourhood_max*pow(self.neighbourhood_min/self.neighbourhood_max, (step_number)/(steps_amount))

    def gauss(self, neuron_number, winner_number):
        return math.exp( -pow( self.neurons[winner_number].distance_measure(self.neurons[neuron_number].weights),2 ) / ( 2*pow(self.neighbourhood,2) ) )

    def weight_update(self, winner_number, one_input):
        for neuron, activity in zip(self.neurons, self.activity_tab):
            if activity == 1:
                if neuron.number != winner_number:
                    gauss = self.gauss(neuron.number, winner_number)
                    new_weight = []
                    for weight, one_in in zip(neuron.weights, one_input):
                        new_weight.append(weight + self.lr * gauss * (one_in - weight))
                    neuron.update_weight(new_weight)
        new_weight = []
        for weight, one_in in zip(self.neurons[winner_number].weights, one_input):
            new_weight.append(weight + self.lr * (one_in - weight))
        self.neurons[winner_number].update_weight(new_weight)

    def dead_update(self, winner_number):
        for neuron, activity in zip(self.neurons, self.activity_tab):
            if activity == 1:
                neuron.potential_update(self.potential_min, winner_number, len(self.neurons))
            else:
                neuron.potential = 1

    def dead_check(self):
        self.activity_tab.clear()
        for neuron in self.neurons:
            if neuron.potential < self.potential_min:
                self.activity_tab.append(0)
            else:
                self.activity_tab.append(1)

    def learn(self, data):
        np.random.shuffle(data)
        for step, one_data in zip(range(1,len(data)+1), data):
            self.lr_neighbourhood_update(step, len(data))
            # self.dead_check()
            winner_number = self.find_winner(one_data)
            self.weight_update(winner_number, one_data)
            # self.dead_update(winner_number)

    def save(self, picture_data, path, name):
        self.activity_tab.clear()
        for i in self.neurons:
            self.activity_tab.append(1)
        name2 = os.path.join(path, str(name) + ".txt")
        newFile = open(name2, "wb")
        tab = []
        for neuron in self.neurons:
            for i in range(len(neuron.weights)):
                tab.append(neuron.get_weight(i))
        for data in picture_data:
            tab.append(self.find_winner(data))
        byte_array = bytearray(tab)
        newFile.write(byte_array)
        newFile.close()

                