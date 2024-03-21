#!/bin/bash

echo "REMEMBER, MUST RUN AS LOCAL USER"

git pull

pushd juxnat_lib
        echo git pull
        git pull
popd

mkdir -p src/juxnat_lib
cp -r juxnat_lib/*.py src/juxnat_lib/
cp *.py src/

echo chmod -R o+rX `pwd`
chmod -R o+rX `pwd`

