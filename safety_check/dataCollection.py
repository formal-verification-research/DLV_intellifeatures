#!/usr/bin/env python

"""
author: Xiaowei Huang
"""
from __future__ import division

import numpy as np
import time
import os
import copy


from configuration import directory_statistics_string

class dataCollection:

    index = 0
    layer = 0
    fileName = os.path.join(directory_statistics_string,'dataCollection.txt')
    fileHandler = 0

    def __init__(self):
        self.runningTime = {}
        self.manipulationPercentage = {}
        self.euclideanDistance = {}
        self.confidence = {}
        self.fileHandler = open(self.fileName, 'w')

    def initialiseIndex(self, index):
        self.index = index

    def initialiseLayer(self, layer):
        self.layer = layer

    def addRunningTime(self, rt):
        self.runningTime[self.index,self.layer] = rt

    def addConfidence(self, cf):
        self.confidence[self.index,self.layer] = cf

    def addManipulationPercentage(self, mp):
        self.manipulationPercentage[self.index,self.layer] = mp

    def addEuclideanDistance(self, eudist):
        self.euclideanDistance[self.index,self.layer] = eudist

    def provideDetails(self):
        self.fileHandler.write("running time: \n")
        for i,r in self.runningTime.items():
            self.fileHandler.write("%s:%s\n"%(i,r))

        self.fileHandler.write("manipulation percentage: \n")
        for i,r in self.manipulationPercentage.items():
            self.fileHandler.write("%s:%s\n"%(i,r))

        self.fileHandler.write("Euclidean distance: \n")
        for i,r in self.euclideanDistance.items():
            self.fileHandler.write("%s:%s\n"%(i,r))

        self.fileHandler.write("confidence: \n")
        for i,r in self.confidence.items():
            self.fileHandler.write("%s:%s\n"%(i,r))
        self.fileHandler.write("\n")

    def summarise(self):
        art = sum(self.runningTime.values()) / len(self.runningTime.values())
        self.fileHandler.write("average running time: %s\n"%(art))
        try:
            amp = sum(self.manipulationPercentage.values()) /float(len(self.manipulationPercentage.values()))
        except ZeroDivisionError:
            amp = 0
        self.fileHandler.write("average manipulation percentage: %s\n"%(amp))
        try:
            eudist = sum(self.euclideanDistance.values()) / len(self.euclideanDistance.values())
        except ZeroDivisionError:
            eudist = 0
        self.fileHandler.write("average euclidean distance: %s\n"%(eudist))

    def close(self):
        self.fileHandler.close()

        
