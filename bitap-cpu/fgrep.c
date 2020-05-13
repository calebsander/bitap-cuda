#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef FUZZY
	#include "fuzzy.h"
#else
	#include "exact.h"
#endif

int main(int argc, char **argv) {
	#ifdef FUZZY
		if (argc < 3) {
			fprintf(stderr, "Usage: %s pattern errors [file ...]\n", argv[0]);
			return 2;
		}
	#else
		if (argc < 2) {
			fprintf(stderr, "Usage: %s pattern [file ...]\n", argv[0]);
			return 2;
		}
	#endif

	char *pattern = argv[1];
	if (strlen(pattern) > PATTERN_LENGTH) {
		fprintf(stderr, "Pattern cannot exceed %d characters\n", PATTERN_LENGTH);
		return 2;
	}

	#ifdef FUZZY
		char *errors_end;
		long errors = strtol(argv[2], &errors_end, 0);
		if (errors_end == argv[2] || *errors_end || errors < 0) {
			fprintf(stderr, "Invalid number of errors: '%s'\n", argv[2]);
			return 2;
		}
	#endif

	// Preprocess the pattern so it can be reused on each line
	pattern_t processed_pattern;
	preprocess_pattern(pattern, &processed_pattern);

	// Process each file
	bool matched = false;
	bool file_error = false;
	#ifdef FUZZY
		char **filename = &argv[3];
	#else
		char **filename = &argv[2];
	#endif
	bool multiple_files = filename[0] && filename[1];
	do {
		// Open the file
		FILE *file;
		if (*filename) {
			file = fopen(*filename, "r");
			if (!file) {
				file_error = true;
				fprintf(stderr, "%s: ", argv[0]);
				perror(*filename);
			}
		}
		else file = stdin;

		// Try to match the pattern against each line
		for (;;) {
			char *line = NULL;
			size_t line_capacity = 0;
			ssize_t line_length = getline(&line, &line_capacity, file);
			if (line_length < 0) {
				free(line);
				break;
			}

			#ifdef FUZZY
				size_t index = find_fuzzy(&processed_pattern, errors, line, line_length);
			#else
				size_t index = find_exact(&processed_pattern, line, line_length);
			#endif
			if (index != NOT_FOUND) {
				matched = true;
				if (multiple_files) fprintf(stdout, "%s:", *filename);
				fwrite(line, sizeof(char), line_length, stdout);
			}
			free(line);
		}
		if (file == stdin) break;

		// Ensure that no error occurred while reading the file
		if (!feof(file)) {
			file_error = true;
			fprintf(stderr, "Failed to read file: ");
			perror(*filename);
		}

		fclose(file);
	} while (*++filename);

	if (file_error) return 3;
	return !matched;
}
