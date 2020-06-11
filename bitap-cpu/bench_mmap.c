#include <assert.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "bench.h"
#include "fuzzy.h"

#define RUNS 100
#define FILENAME "../oanc.txt"
#define PATTERN "throughout"
#define ERRORS 2

size_t last_line_start(const char *buffer, size_t length) {
	while (length) {
		asm(""); // prevent this being optimized out
		length--;
		if (buffer[length] == EOL) return length + 1;
	}
	return 0;
}

int main() {
	int fd = open(FILENAME, O_RDONLY);
	assert(fd > 0);
	struct stat file_stats;
	int result = fstat(fd, &file_stats);
	size_t length = file_stats.st_size;
	assert(!result);
	start_time(LOAD_FILE);
	char *text = mmap(NULL, length, PROT_READ, MAP_PRIVATE, fd, 0);
	assert(text != MAP_FAILED);
	stop_time();
	result = close(fd);
	assert(!result);

	for (size_t runs = RUNS; runs > 0; runs--) {
		pattern_t processed_pattern;
		preprocess_pattern(PATTERN, &processed_pattern);

		start_time(ALLOCATE_INDICES);
		size_t *match_indices = malloc(sizeof(size_t[length]));
		assert(match_indices);
		stop_time();

		size_t match_count;
		find_fuzzy(&processed_pattern, ERRORS, text, length, match_indices, &match_count);

		start_time(FIND_LINES);
		size_t last_index = 0;
		for (size_t i = 0; i < match_count; i++) {
			size_t match_index = match_indices[i];
			if (match_index < last_index) continue;

			last_line_start(text, match_index);
			char *line_end = memchr(&text[match_index], EOL, length - match_index);
			last_index = line_end ? (size_t) (line_end - text) + 1 : length;
		}
		stop_time();

		free(match_indices);
	}

	result = munmap(text, length);
	assert(!result);
	print_bench_times();
}
