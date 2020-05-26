#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef FUZZY
	#include "fuzzy.h"
#else
	#include "exact.h"
#endif

#define BUFFER_SIZE (128 << 10) // process 128 KB at a time

// Returns the first index of the last line in a buffer with the given start and length
size_t last_line_start(char *buffer, size_t length) {
	while (length) {
		length--;
		if (buffer[length] == EOL) return length + 1;
	}
	return 0;
}

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
	if (!*pattern) {
		fputs("Pattern cannot be empty\n", stderr);
		return 2;
	}
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

	// Allocate input buffer and array of matches
	char *buffer = malloc(sizeof(char[BUFFER_SIZE]));
	size_t *match_indices = malloc(sizeof(size_t[BUFFER_SIZE]));
	if (!(buffer && match_indices)) {
		fputs("Failed to allocate buffer\n", stderr);
		return 3;
	}

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
		size_t filename_length;
		if (*filename) {
			file = fopen(*filename, "r");
			if (!file) {
				file_error = true;
				fprintf(stderr, "%s: ", argv[0]);
				perror(*filename);
			}
			filename_length = strlen(*filename);
		}
		else file = stdin;

		// Try to match the pattern against each chunk of the file
		size_t buffer_kept = 0;
		for (;;) {
			// We keep `buffer_kept` characters from the last chunk, then fill the buffer
			size_t buffer_left = BUFFER_SIZE - buffer_kept;
			// TODO use `read()` syscall to avoid waiting for user input on stdin
			size_t buffer_read = fread(&buffer[buffer_kept], sizeof(char), buffer_left, file);
			if (ferror(file)) {
				file_error = true;
				fprintf(stderr, "Failed to read %s: ", argv[0]);
				perror(*filename);
				break;
			}

			// Compute the total size of the buffer; if it's empty, we are done
			size_t buffer_size = buffer_kept + buffer_read;
			if (!buffer_size) break;

			// If some new characters were read, only search through the last full line
			if (buffer_read) {
				buffer_kept = buffer_size - last_line_start(buffer, buffer_size);
				buffer_size -= buffer_kept;
			}
			// Otherwise, the file is missing a trailing newline, so read the entire file
			else buffer_kept = 0;

			// Find all matches in the buffer
			size_t match_count;
			#ifdef FUZZY
				find_fuzzy(
					&processed_pattern, errors,
					buffer, buffer_size,
					match_indices, &match_count
				);
			#else
				find_exact(&processed_pattern, buffer, buffer_size, match_indices, &match_count);
			#endif

			// Print out the line containing each match
			size_t last_index = 0;
			for (size_t i = 0; i < match_count; i++) {
				// If this line was already printed, skip it
				size_t match_index = match_indices[i];
				if (match_index < last_index) continue;

				matched = true;
				size_t line_start = last_line_start(buffer, match_index);
				char *line_end = memchr(&buffer[match_index], '\n', buffer_size - match_index);
				last_index = line_end ? line_end - buffer + 1 : buffer_size;
				if (multiple_files) {
					// Print out the filename as well if multiple files were searched
					fwrite(*filename, sizeof(char), filename_length, stdout);
					putchar(':');
				}
				fwrite(&buffer[line_start], sizeof(char), last_index - line_start, stdout);
			}

			// Save the unconsumed part of the last line
			memmove(buffer, &buffer[buffer_size], sizeof(char[buffer_kept]));
		}
		if (!*filename) break;

		fclose(file);
	} while (*++filename);

	free(buffer);
	free(match_indices);

	if (file_error) return 3;
	return !matched;
}
