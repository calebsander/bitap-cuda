#include "pattern.h"
#include <assert.h>
#include <stdbool.h>
#include <string.h>
#include "bench.h"

// Sets or unsets a bit of a character's pattern mask
void update_char(
	pattern_t *pattern,
	unsigned char c,
	bool add,
	pattern_mask_t index_mask
) {
	pattern_mask_t *char_mask = &pattern->char_masks[c];
	if (add) *char_mask |= index_mask;
	else *char_mask &= ~index_mask;
}
// Updates a range of characters' pattern masks
void update_range(
	pattern_t *pattern,
	unsigned char start,
	unsigned char end,
	bool add,
	pattern_mask_t index_mask
) {
	for (size_t i = start; i <= end; i++) {
		// EOL character must never match so we don't match across lines
		if (i != EOL) update_char(pattern, i, add, index_mask);
	}
}
// Sets a given index in all characters' pattern masks
void include_all_chars(pattern_t *pattern, pattern_mask_t index_mask) {
	update_range(pattern, 0, -1, true, index_mask);
}
// Converts a regex escape sequence to the character it represents.
// `c` is the character after the backslash.
char read_escaped_char(char c) {
	switch (c) {
		case 'a': return '\a';
		case 'e': return '\e';
		case 'f': return '\f';
		case 'r': return '\r';
		case 't': return '\t';
		case '.':
		case '[':
		case ']':
		case '^':
		case '-':
		case '\\': return c;
		default: assert(false);
	}
}
// Sets or unsets an escape sequence (e.g. "\d") at the given index
void update_escaped(pattern_t *pattern, char c, bool add, pattern_mask_t index_mask) {
	switch (c) {
		case 'd':
			update_range(pattern, '0', '9', add, index_mask);
			break;
		case 's':
			update_char(pattern, ' ', add, index_mask);
			update_char(pattern, '\t', add, index_mask);
			update_char(pattern, '\r', add, index_mask);
			update_char(pattern, '\f', add, index_mask);
			break;
		case 'w':
			update_range(pattern, 'A', 'Z', add, index_mask);
			update_range(pattern, 'a', 'z', add, index_mask);
			update_range(pattern, '0', '9', add, index_mask);
			update_char(pattern, '_', add, index_mask);
			break;
		default:
			update_char(pattern, read_escaped_char(c), add, index_mask);
	}
}

void preprocess_pattern(const char *pattern, pattern_t *processed_pattern) {
	// Empty patterns cause problems and don't seem useful
	assert(*pattern);
	start_time(PROCESS_PATTERN);

	// Construct bitmasks representing where each character appears in `pattern`.
	// Bit `i` of `char_masks[c]` is set iff `c` occurs at index `i + 1` of `pattern`.
	memset(&processed_pattern->char_masks, 0, sizeof(processed_pattern->char_masks));
	size_t pattern_index = 0;
	pattern_mask_t index_mask = 1, last_index_mask = 0;
	char c;
	while ((c = *pattern)) {
		switch (c) {
			// An escape sequence or shorthand character class, e.g. "\n" or "\d"
			case '\\':
				update_escaped(processed_pattern, *++pattern, true, index_mask);
				break;
			// Matches any character
			case '.':
				include_all_chars(processed_pattern, index_mask);
				break;
			// A character class
			case '[':
				c = *++pattern;
				bool add = c != '^'; // ^ negates the character class
				if (!add) {
					include_all_chars(processed_pattern, index_mask);
					c = *++pattern;
				}
				while (c != ']') {
					assert(c);
					bool is_escaped = c == '\\';
					if (is_escaped) pattern++;
					// Check if this is a range, e.g. "[a-z]"
					char range_end;
					if (pattern[1] == '-' && (range_end = pattern[2]) && range_end != ']') {
						// If it is a range, find its bounds (the start or end bounds may be escaped)
						if (is_escaped) c = read_escaped_char(*pattern);
						pattern += 2;
						if (range_end == '\\') range_end = read_escaped_char(*++pattern);
						// Set or unset this range of characters
						update_range(processed_pattern, c, range_end, add, index_mask);
					}
					else {
						// Set or unset a character or shorthand character class (e.g. "[\d]")
						if (is_escaped) update_escaped(processed_pattern, *pattern, add, index_mask);
						else update_char(processed_pattern, c, add, index_mask);
					}
					c = *++pattern;
				}
				break;
			// A normal character: add it to the given regex index
			default:
				update_char(processed_pattern, c, true, index_mask);
		}
		pattern_index++;
		last_index_mask = index_mask;
		index_mask <<= 1;
		pattern++;
	}

	// `pattern_mask_t` must have enough bits to represent each character of `pattern`
	assert(pattern_index <= sizeof(pattern_mask_t) * BYTE_BITS);
	processed_pattern->length = pattern_index;
	processed_pattern->end_mask = last_index_mask;
	stop_time();
}
