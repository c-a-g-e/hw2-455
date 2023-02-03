# Homework 2
# name: Martine De Cock
# description: Training and testing decision trees with discrete-values attributes

import sys
import math
import numpy as np
import pandas as pd


class DecisionNode:

    # A DecisionNode contains an attribute and a dictionary of children. 
    # The attribute is either the attribute being split on, or the predicted label if the node has no children.
    def __init__(self, attribute):
        self.attribute = attribute
        self.children = {}

    # Visualizes the tree
    def display(self, level=0):
        if self.children == {}:  # reached leaf level
            print(": ", self.attribute, end="")
        else:
            for value in self.children.keys():
                prefix = "\n" + " " * level * 4
                print(prefix, self.attribute, "=", value, end="")
                self.children[value].display(level + 1)

    # Predicts the target label for instance x
    def predicts(self, x):
        if self.children == {}:  # reached leaf level
            return self.attribute
        value = x[self.attribute]
        subtree = self.children[value]
        return subtree.predicts(x)


def entropy_by_attribute(attribute, target):
    target_variables = train[target].unique()
    variables = train[attribute].unique()
    total_entropy = 0

    for variable in variables:
        variable_entropy = 0
        for target_variable in target_variables:
            find_variable = train[attribute][train[attribute] == variable]
            numerator = len(find_variable[train[target] == target_variable])
            denominator = len(find_variable)
            p = numerator/denominator
            if p == 0:
                variable_entropy += 0
            else:
                variable_entropy += -(numerator/denominator) * np.log2(p)
        sv_fraction = denominator / len(train)
        total_entropy += -sv_fraction * variable_entropy
    return abs(total_entropy)


def gain(attribute, target):
    train_entropy = 0
    variables = train[target].unique()
    for variable in variables:
        p = train[target].value_counts()[variable] / len(train[target])
        train_entropy += -p * np.log2(p)

    return train_entropy - entropy_by_attribute(attribute, target)


def id3(examples, target, attributes):

    if len(examples[target].unique()) == 1:
        if examples[target].unique()[0] == 'yes' or examples[target].unique()[0] == 1:
            return DecisionNode('yes')
        else:
            return DecisionNode('no')

    if len(attributes) == 0:
        return DecisionNode(examples[target].unique().max())

    max_gain = 0
    highest_gain_attribute = ''
    for attribute in attributes:
        current_gain = gain(attribute, target)
        if current_gain > max_gain:
            max_gain = current_gain
            highest_gain_attribute = attribute

    attributes.remove(highest_gain_attribute)
    root_node = DecisionNode(highest_gain_attribute)
    attr_values = examples[highest_gain_attribute].unique()
    for value in attr_values:
        value_subset = examples[examples[highest_gain_attribute] == value]
        if len(value_subset) == 0:
            root_node.children[value] = DecisionNode(examples[target].unique().max())
        else:
            root_node.children[value] = id3(value_subset, target, attributes)

    return root_node


# Illustration of functionality of DecisionNode class
def funTree():
    myLeftTree = DecisionNode('humidity')
    myLeftTree.children['normal'] = DecisionNode('no')
    myLeftTree.children['high'] = DecisionNode('yes')
    myTree = DecisionNode('wind')
    myTree.children['weak'] = myLeftTree
    myTree.children['strong'] = DecisionNode('no')
    return myTree


####################   MAIN PROGRAM ######################

# Reading input data
train = pd.read_csv(sys.argv[1])
test = pd.read_csv(sys.argv[2])
target = sys.argv[3]
attributes = train.columns.tolist()
attributes.remove(target)

# Learning and visualizing the tree
tree = id3(train, target, attributes)
tree.display()

# Evaluating the tree on the test data
correct = 0
for i in range(0, len(test)):
    if str(tree.predicts(test.loc[i])) == str(test.loc[i, target]):
        correct += 1
print("\nThe accuracy is: ", correct / len(test))
