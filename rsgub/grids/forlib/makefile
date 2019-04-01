export FC=gfortran
export CC=gcc

mods=fill_qk

all:
	for name in $(mods); do \
		f2py -c -m $$name $$name.f90; \
	done

clean:
	for name in $(mods); do \
	  rm $$name.so; \
	done
