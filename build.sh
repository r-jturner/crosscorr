#!/bin/bash

set -e

cd csrc
make

mkdir -p ../src/crosscorr/obj

cp obj/corrdesi.so \
	../src/crosscorr/obj/
