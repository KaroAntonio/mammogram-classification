#!/bin/bash

# entry point for running the scoring submission

# this script (or a Python script that this script calls) must output a
# tsv file in /output/predictions.tsv with the following format
# subjectId      laterality      confidence
# 1                 L              0.01
# 1                 R              0.05
# 2                 L              0.00
# 2                 R              0.01

python score_mammo.py
