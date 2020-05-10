PATTERN_LENGTH = 64

CC = clang
CFLAGS = -Wall -Wextra -O3 -DPATTERN_LENGTH=$(PATTERN_LENGTH)

ifdef DEBUG
	CFLAGS  += -fsanitize=address,undefined
	LDFLAGS += -fsanitize=address,undefined
endif

fgrep: fgrep.o exact.o

exact.o: exact.c exact.h
fgrep.o: exact.h

clean:
	rm -f *.o fgrep
