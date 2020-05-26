#include "bench.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "fuzzy.h"

#define NS_PER_SEC ((duration_t) 1e9)

duration_t get_duration(struct timespec *start, struct timespec *end) {
	return (end->tv_sec - start->tv_sec) * NS_PER_SEC + end->tv_nsec - start->tv_nsec;
}

// TODO: avoid duplicating this with fgrep.c
size_t last_line_start(const char *buffer, size_t length) {
	while (length) {
		length--;
		if (buffer[length] == EOL) return length + 1;
	}
	return 0;
}

times_t __attribute__((optnone)) benchmark(
	const char *pattern,
	size_t errors,
	const char *text,
	size_t text_length
) {
	times_t times;
	struct timespec start, end;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
	pattern_t processed_pattern;
	preprocess_pattern(pattern, &processed_pattern);
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end);
	times.pattern_processing = get_duration(&start, &end);

	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
	size_t *match_indices = malloc(sizeof(size_t[text_length]));
	assert(match_indices);
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end);
	times.index_allocation = get_duration(&start, &end);

	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
	size_t match_count;
	find_fuzzy(&processed_pattern, errors, text, text_length, match_indices, &match_count);
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end);
	times.fuzzy_match = get_duration(&start, &end);

	// We don't have to sort the indices since they were generated in order
	times.sort_matches = 0;

	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
	size_t last_index = 0;
	for (size_t i = 0; i < match_count; i++) {
		size_t match_index = match_indices[i];
		if (match_index < last_index) continue;

		size_t line_start = last_line_start(text, match_index);
		(void) line_start;
		char *line_end = memchr(&text[match_index], '\n', text_length - match_index);
		last_index = line_end ? line_end - text + 1 : text_length;
	}
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end);
	times.find_lines = get_duration(&start, &end);

	free(match_indices);
	return times;
}

times_t average_times(const times_t *times, size_t length) {
	assert(length);

	times_t sum = {0};
	for (size_t i = 0; i < length; i++) {
		sum.pattern_processing += times[i].pattern_processing;
		sum.index_allocation += times[i].index_allocation;
		sum.fuzzy_match += times[i].fuzzy_match;
		sum.sort_matches += times[i].sort_matches;
		sum.find_lines += times[i].find_lines;
	}
	return (times_t) {
		.pattern_processing = sum.pattern_processing / length,
		.index_allocation = sum.index_allocation / length,
		.fuzzy_match = sum.fuzzy_match / length,
		.sort_matches = sum.sort_matches / length,
		.find_lines = sum.find_lines / length,
	};
}
