#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "bench.h"

#define RUNS 100
#define FILENAME "oanc.txt"
#define PATTERN "throughout"
#define ERRORS 2

int main() {
	FILE *file = fopen(FILENAME, "r");
	assert(file);
	fseek(file, 0, SEEK_END);
	size_t length = ftell(file);
	rewind(file);
	char *text = malloc(length);
	assert(text);
	size_t read = fread(text, 1, length, file);
	assert(read == length);
	fclose(file);

	times_t bench_times[RUNS];
	for (size_t i = 0; i < RUNS; i++) {
		bench_times[i] = benchmark(PATTERN, ERRORS, text, length);
	}
	times_t average = average_times(bench_times, RUNS);
	printf("pattern_processing: %e ns\n", (double) average.pattern_processing);
	printf("index_allocation: %e ns\n", (double) average.index_allocation);
	printf("fuzzy_match: %e ns\n", (double) average.fuzzy_match);
	printf("sort_matches: %e ns\n", (double) average.sort_matches);
	printf("find_lines: %e ns\n", (double) average.find_lines);
}
