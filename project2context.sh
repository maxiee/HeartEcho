#!/bin/bash

python ../ProjectToContext/main.py app/lib

# 从 ./app 目录中，读取 sources.txt，存入一个变量，并删除 sources.txt 文件
sources1=$(cat ./app/sources.txt)
rm ./app/sources.txt

python ../ProjectToContext/main.py backend

# 从当前目录下，读取 sources.txt，存入一个变量，并删除 sources.txt 文件
sources2=$(cat ./sources.txt)
rm ./sources.txt

# 读取当前目录下的 README.md 文件，存入一个变量
readme=$(cat ./README.md)

# 合并 sources1 和 sources2（连接处插入换行），存入 sources.txt
echo -e "$readme\n$sources1\n$sources2" > ./sources.txt