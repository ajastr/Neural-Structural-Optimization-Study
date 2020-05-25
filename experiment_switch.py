from training_switch import train_all
from neural_structural_optimization import problems
import os
import pandas
import csv
import numpy as np
import re
from neural_structural_optimization.problems import PROBLEMS_BY_NAME

examples = PROBLEMS_BY_NAME
example_list =[]
example_names =list(examples)

for i in example_names:
    size1=re.findall("([0-9]*)x",str(i))[0]
    size2=re.findall("x([0-9]*)",str(i))[0]
    if int(size1)<200 and int(size2)<200:
        example_list.append(i)
print(example_list)

#rng=np.random.RandomState(777)
#example_list = rng.choice(list(examples),15,replace=False)

max_iterations=200
switch = [1,2,4,8,16,32,64]
width = [128,64,32,16,1]
cnn_kwargs=dict(resizes=(1, 1, 2, 2, 1))

with open('results/training_switch.csv', mode='w') as training_data:
    training_writer = csv.writer(training_data, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
    training_writer.writerow(['Switch','Example','Loss CNN switch',"Loss CNN max iterations",'Loss Pixel Switch','Loss Pixel No Switch'])

def summary(examples, cnn_layers, switch,cnn_kwargs):

    for name in example_list:
        example = examples[name]
        example_summary={}
        example_relative = {}

        for i in switch:

            example_losses = train_all(example=example,switch=i,name=name,max_iterations=max_iterations,cnn_layers=width, cnn_kwargs=cnn_kwargs)

            with open('results/training_switch.csv', mode='a') as training_data:
                training_writer = csv.writer(training_data, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
                training_writer.writerow([i,name,example_losses[0],example_losses[1],example_losses[2],example_losses[3]])

summary(examples,width,switch,cnn_kwargs)
