include ../make.inc

CXXFLAGS+=-I${RACE_INCDIR} -I./include/ -std=c++17

OBJ=sparsemat.o densemat.o mpk.o

lib/libmpk.so: ${OBJ} Makefile ../make.inc
	${CXX} ${CXXFLAGS} -shared -o lib/libmpk.so ${OBJ} -L${PWD}/lib -lhwloc -L${RACE_LIBDIR} -lRACE -Wl,-rpath=${PWD}/lib

clean:
	rm lib/libmpk.so *.o
