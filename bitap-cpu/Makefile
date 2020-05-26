PATTERN_LENGTH = 64

CC = clang
CFLAGS = -Wall -Wextra -O3 -DPATTERN_LENGTH=$(PATTERN_LENGTH)

ifdef DEBUG
	CFLAGS  += -fsanitize=address,undefined -g
	LDFLAGS += -fsanitize=address,undefined
endif

EXECUTABLES = agrep fgrep bench-oanc

all: $(EXECUTABLES)

agrep: agrep.o fuzzy.o pattern.o
fgrep: fgrep.o exact.o pattern.o
fuzz-%: fuzz-%.c exact.o pattern.o
	$(CC) $(CFLAGS) -fsanitize=fuzzer $^ -o $@
bench-oanc: fuzzy.o bench.o pattern.o

bench.o: bench.h
bench-oanc.o: bench.h
exact.o: exact.h pattern.h
fuzzy.o: fuzzy.h pattern.h
pattern.o: pattern.h

agrep.o: fgrep.c fuzzy.h pattern.h
	$(CC) $(CFLAGS) -DFUZZY -c $< -o $@
fgrep.o: exact.h pattern.h

clean:
	rm -f *.o $(EXECUTABLES) fuzz-literal-exact
