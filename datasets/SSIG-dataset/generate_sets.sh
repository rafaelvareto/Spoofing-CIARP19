#!/bin/bash
for foldername in `ls -1`
do
    for filename in `ls $foldername/*.MP4`
    do
        echo $filename $foldername
    done
done 
