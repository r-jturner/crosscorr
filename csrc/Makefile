IDIR =../include
CC?=gcc
CFLAGS=-I$(IDIR) -fPIC -O2 -Wall -Werror -fopenmp

ODIR=obj
SODIR=../src/crosscorr/obj

_DEPS = corr_desi.h
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

_OBJ = corr_desi.o corr_desi_smu.o corr_desi_xyz.o
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

.DEFAULT_GOAL := all

all: $(SODIR)/corrdesi.so

$(ODIR)/%.o: %.c $(DEPS)
	mkdir -p $(ODIR)
	$(CC) -c -o $@ $< $(CFLAGS)

$(SODIR)/corrdesi.so: $(OBJ)
	mkdir -p $(SODIR)
	$(CC) -shared -o $@ $^ $(CFLAGS)

.PHONY: clean rebuild install

clean:
	rm -f $(ODIR)/*.o
	rm -f $(SODIR)/*.so
	rm -f *~
	rm -f $(IDIR)/*~

rebuild: clean all

install: all
	pip install -e ..
