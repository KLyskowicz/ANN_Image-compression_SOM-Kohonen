import random
import numpy as np
import math 

class Neuron(object):               # 2 wymiarowa tablica [min_value, max_value]
    def __init__(self, inputs_amount, input_min_max_value, number):
        self.potential = 1
        self.weights = np.zeros(inputs_amount)
        self.number = number
        for i, value in zip(range(inputs_amount), input_min_max_value):
            self.weights[i] = random.uniform(value[0],value[1])

    def distance_measure(self, point):
        summation = 0
        for i, j in zip(self.weights, point):
            summation += (i-j)*(i-j)
        return math.sqrt(summation)
    
    def potential_update(self, potential_min, winner_number, neuron_amount):
        if self.number != winner_number:
            self.potential += 1/neuron_amount
        else:
            self.potential -= potential_min

    def update_weight(self, new_weight):
        for i, nw in zip(range(len(self.weights)), new_weight):
            self.weights[i] = nw

    def get_weight(self, number):
        return int(round(self.weights[number]))
