#!/bin/bash
cd "$(dirname "$0")"
source myenv/bin/activate
./fs4.py
aplay out.wav
