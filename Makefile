CC=/usr/bin/gcc
CFLAGS += -O3 -march=native -fomit-frame-pointer
#LDFLAGS=-lcrypto

SOURCES= brv.cpp getw.cpp ntt.cpp main.cpp Pt_3.cpp
HEADERS= brv.h getw.h ntt.h params.h Pt_3.h
main: $(HEADERS) $(SOURCES)
	$(CC) $(CFLAGS) -o $@ $(SOURCES) $(LDFLAGS)

.PHONY: clean

clean:
	-rm main

