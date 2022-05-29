import requests
from flask import Flask, render_template, request

app = Flask(__name__)

import pandas as pd
import math
from collections import Counter

data = pd.DataFrame(pd.read_csv("testData.csv"))

catColNames = "fabric_type,brand,main_colour".split(',')
# conColNames = "price".split(',')

# preference = {"fabric_type": {"nylon": 4, "silk": 2, "wool": 1},
#               "brand": {"nighkey": 2, "soupreme": 1},
#               "main_colour": {"red": 2, "blue": 1},
#               "price": 20}

linWeight = 5

weights = {"fabric_type": 1,
           "brand": 1,
           "main_colour": 1,
           }  # "price": 1

nTestRounds = 10

nRows = len(data[data.columns[0]])
rollingAvgSize = 10

choiceHistory = {"fabric_type": [],
                 "brand": [],
                 "main_colour": [],
                 }  # "price": []


def get_preference(rollingAvgSize, idx, chosen):
    entry = data.iloc[idx].to_dict()

    if chosen:
        for col in catColNames:
            choiceHistory[col].append(entry[col])
            choiceHistory[col] = choiceHistory[col][::-1][:rollingAvgSize]
        # for col in conColNames:
        #    choiceHistory[col].append(entry[col])
    else:
        for col in catColNames:
            if entry[col] in choiceHistory[col]:
                del choiceHistory[col][choiceHistory[col].index(entry[col])]

    preference = {}

    for col in catColNames:
        preference[col] = Counter(choiceHistory[col])

    # for col in conColNames:
    #    preference[col] = sum(choiceHistory[col][len(choiceHistory[col]) - rollingAvgSize:]) / rollingAvgSize

    return preference


def get_score(preference, target):
    score = 0

    for col in catColNames:
        if target[col] in preference[col].keys():
            score += (rollingAvgSize - preference[col][target[col]]) * weights[col]

    # sum = 0

    # for col in conColNames:
    #    sum += ((target[col] - preference[col]) ** 2) * weights[col]

    # score += linWeight / math.sqrt(sum)

    return score


def get_most_relevant_prod(currPreference, indices):
    scores = []

    for x in indices:
        b = data.iloc[x].to_dict()
        scores.append(get_score(currPreference, b))

    scores, indices = zip(*sorted(zip(scores, indices)))

    return indices[::-1][0]


cart = []
prod_idx = 0
inds = list(range(nTestRounds + 1, nRows))
current_preference = {}


@app.route('/start', methods=['POST'])
@app.route('cardScreen/card', methods=['POST'])
def swipe():
    # Might need to write accessors/mutators for vars because of scope issues
    # TODO get direction somehow
    direction = ''

    if prod_idx <= 10:

        if direction in ["UP, RIGHT"]:
            t = get_preference(rollingAvgSize, prod_idx, chosen=True)
            if direction == "UP":
                cart.append(prod_idx)
        else:
            t = get_preference(rollingAvgSize, prod_idx, chosen=False)

        prod_idx += 1


        if prod_idx == 10:
            current_preference = t

    if prod_idx >= 10:
        if len(inds) > 0:

            if direction in ["UP, RIGHT"]:
                current_preference = get_preference(rollingAvgSize, prod_idx, chosen=True)
                if direction == "UP":
                    cart.append(prod_idx)
            else:
                current_preference = get_preference(rollingAvgSize, prod_idx, chosen=False)

            prod_id = get_most_relevant_prod(current_preference, inds)

            del inds[inds.index(prod_id)]

    # TODO render card for new prod_idx


# def main():
#     for x in range(nTestRounds - 1):
#         get_preference(rollingAvgSize, x)
#
#     current_preference = get_preference(rollingAvgSize, nTestRounds)
#
#     inds = list(range(nTestRounds + 1, nRows))
#
#     x = 0
#     while len(inds) > 0:
#         prod_id = get_most_relevant_prod(current_preference, inds)
#         current_preference = get_preference(rollingAvgSize, prod_id)
#
#         del inds[inds.index(prod_id)]
#         x += 1
#
#
# main()
