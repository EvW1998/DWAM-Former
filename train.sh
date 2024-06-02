#!/bin/bash

for num in {1..1}
do
  rm -r experiments/*
#  rm -r result/*

  python train_model.py -s 123

#  cat result/*
#
#  file="upload_result/sf-result-$(date "+%Y%m%d-%H%M%S").zip"
#
#  zip "${file}" -r experiments/ result/

done

#file="packed-result-$(date "+%Y%m%d-%H%M%S").zip"
#zip "${file}" -r upload_result/
#oss cp "${file}" oss://
#rm "${file}"
#rm -r upload_result/*
#
#if [ "$1" = "shutdown" ]; then
#  shutdown
#fi
