include ../make.inc

OBJ=mkl_interface.o

lib/libmpk.so: ${OBJ} Makefile ../make.inc
	${CC} ${CFLAGS} -shared ${OBJ} -o lib/libmkl.so -Wl,--whole-archive ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--no-whole-archive

%.o: %.c Makefile make.inc
	${CC} ${CFLAGS} -c $<
    
clean:
	rm -f lib/libmkl.so *.o