#!/bin/bash

#folds="fold1 fold2"
folds="fold3"
wav_dir="audio"

for fold in $folds; do
	save_dir="$wav_dir/${fold}_dwnsmp"
	mkdir $save_dir
	search_dir="$wav_dir/$fold"
	files=$(find $search_dir -type f -name "*.wav")
	for fn in $files; do
	    echo "Process $fn ..."		
	    name=$(basename $fn)
		sox -V $fn -c 1 -t wav -b 16 -r 22050 $save_dir/$name
	done
done
