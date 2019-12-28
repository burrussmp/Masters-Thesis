#!/bin/bash
i="0"

while [ $i -lt 7e ]
do
python3 heatmap.py
i=$[$i+1]
done
