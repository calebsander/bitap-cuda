#ifndef BENCH_H
#define BENCH_H

#include <stddef.h>
#include <stdint.h>

typedef uint64_t duration_t;

typedef struct {
	duration_t pattern_processing;
	duration_t index_allocation;
	duration_t fuzzy_match;
	duration_t sort_matches;
	duration_t find_lines;
} times_t;

times_t benchmark(
	const char *pattern,
	size_t errors,
	const char *text,
	size_t text_length
);

times_t average_times(const times_t *times, size_t length);

#endif // #ifndef BENCH_H
