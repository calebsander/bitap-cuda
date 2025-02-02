PATTERN_LENGTH = 32

CC = gcc
CFLAGS = -Wall -Wextra -O3 -DPATTERN_LENGTH=$(PATTERN_LENGTH)

ifdef DEBUG
	CFLAGS  += -fsanitize=address,undefined -g
	LDFLAGS += -fsanitize=address,undefined
endif

EXECUTABLES = agrep fgrep bench_mmap bench_read

all: $(EXECUTABLES)

agrep: agrep.o fuzzy.o pattern.o
fgrep: fgrep.o exact.o pattern.o
fuzz_%: fuzz_%.c exact.o pattern.o # requires clang
	$(CC) $(CFLAGS) -fsanitize=fuzzer $^ -o $@
bench_mmap: bench_mmap.c bench.c fuzzy.c pattern.c
	$(CC) $(CFLAGS) -DBENCH $^ -lm -o $@
bench_read: bench_read.c bench.c fuzzy.c pattern.c
	$(CC) $(CFLAGS) -DBENCH $^ -lm -o $@

bench.o: bench.h
exact.o: exact.h pattern.h
fuzzy.o: fuzzy.h pattern.h
pattern.o: pattern.h

agrep.o: fgrep.c fuzzy.h pattern.h
	$(CC) $(CFLAGS) -DFUZZY -c $< -o $@
fgrep.o: exact.h pattern.h

clean:
	rm -f *.o $(EXECUTABLES) fuzz_literal_exact
