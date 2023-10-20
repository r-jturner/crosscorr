IDIR =../include
CC=gcc
CFLAGS=-I$(IDIR) -fPIC -O2 -Wall -Werror -fopenmp

ODIR=obj

_DEPS = corr_desi.h
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

_OBJ = corr_desi.o corr_desi_smu.o corr_desi_xyz.o
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))


$(ODIR)/%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

$(ODIR)/corrdesi.so: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) -shared

.PHONY: clean

clean:
	rm -f $(ODIR)/*.o *~ core $(IDIR)/*~
	rm -f $(ODIR)/*.so *~ core $(IDIR)/*~ 
