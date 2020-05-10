#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include "exact.h"

int main(int argc, char **argv) {
	if (argc < 2) {
		fprintf(stderr, "Usage: %s pattern [file ...]\n", argv[0]);
		return 2;
	}

	// Preprocess the pattern so it can be reused on each line
	char *pattern = argv[1];
	pattern_t processed_pattern;
	preprocess_pattern(pattern, &processed_pattern);

	// Process each file
	bool matched = false;
	bool file_error = false;
	bool multiple_files = argc > 3;
	char **filename = &argv[2];
	while (*filename) {
		// Open the file
		FILE *file = fopen(*filename, "r");
		if (!file) {
			file_error = true;
			fprintf(stderr, "%s: ", argv[0]);
			perror(*filename);
		}

		// Try to match the pattern against each line
		for (;;) {
			char *line = NULL;
			size_t line_capacity = 0;
			ssize_t line_length = getline(&line, &line_capacity, file);
			if (line_length < 0) break;

			if (find_exact(&processed_pattern, line, line_length) != NOT_FOUND) {
				matched = true;
				if (multiple_files) fprintf(stdout, "%s:", *filename);
				fwrite(line, sizeof(char), line_length, stdout);
			}
			free(line);
		}

		// Ensure that no error occurred while reading the file
		if (!feof(file)) {
			file_error = true;
			fprintf(stderr, "Failed to read file: ");
			perror(*filename);
		}
		fclose(file);
		filename++;
	}

	if (file_error) return 3;
	return !matched;
}
