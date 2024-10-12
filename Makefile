
include make.inc

OBJ=vector_kernels.o spmv_kernels.o

libkernels.so: ${OBJ} Makefile make.inc
	${CC} ${CFLAGS} -shared -o libkernels.so ${OBJ}

%.o: %.c Makefile make.inc
	${CC} ${CFLAGS} -c $<

clean:
	rm -f *.o libkernels.so
