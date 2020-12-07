# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 18:33:56 2020

@author: cheng
"""
import os

def mak_dir():
    # Make all the folders to save the intermediate results
    # ToDo chenge this to make compatible with linus
    model_dir = "../models"
    processed_train = "../processed_data/train"
    processed_challenge = "../processed_data/challenge"
    # Save the cvae model's prediction
    prediction = "../prediction"
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
        print('%s created'%model_dir)
    if not os.path.exists(processed_train):
        os.mkdir(processed_train)
        print('%s created'%processed_train)
    if not os.path.exists(processed_challenge):
        os.mkdir(processed_challenge)
        print('%s created'%processed_challenge)
    if not os.path.exists(prediction):
        os.mkdir(prediction)
        print('%s created'%prediction)

