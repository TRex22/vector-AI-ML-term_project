#!/bin/bash
rm data/*.npz
python tf_nn1.py
python tf_nn3.py

python encode_3_3.py