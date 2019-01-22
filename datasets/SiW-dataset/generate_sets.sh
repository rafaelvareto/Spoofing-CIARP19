#!/bin/bash
for foldername in `ls -1`
do
    for filename in `ls $foldername/*.avi`
    do
        echo $filename $foldername
    done
done 
