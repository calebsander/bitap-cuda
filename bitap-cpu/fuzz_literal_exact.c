#include <assert.h>
#include <string.h>
#include "exact.h"

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
	if (!size) return 0;
	uint8_t pattern_length = data[0];
	if (pattern_length > PATTERN_LENGTH) return 0;
	data++;
	size--;

	if (size < pattern_length) return 0;
	// Ensure the pattern doesn't contain special characters
	if (
		memchr(data, '\\', pattern_length) ||
		memchr(data, '.', pattern_length) ||
		memchr(data, '[', pattern_length)
	) return 0;
	char pattern[pattern_length + 1];
	memcpy(pattern, data, pattern_length);
	pattern[pattern_length] = '\0';
	data += pattern_length;
	size -= pattern_length;

	if (!size) return 0;
	uint8_t text_length = data[0];
	data++;
	size--;

	if (size < text_length) return 0;
	const char *text = (const char *) data;
	// Ensure the text doesn't contain a premature null terminator
	char *text_end = memchr(text, '\0', text_length);
	if (text_end) text_length = text_end - text;
	char *pattern_match = strnstr(text, pattern, text_length);
	pattern_t processed_pattern;
	preprocess_pattern(pattern, &processed_pattern);
	size_t index = find_exact(&processed_pattern, text, text_length);
	assert(index == (pattern_match ? pattern_match - text : NOT_FOUND));
	return 0;
}
