#include "exact.h"

void find_exact(
	const pattern_t *pattern,
	const char *text,
	size_t text_length,
	size_t *match_indices,
	size_t *match_count
) {
	*match_count = 0;

	/* Bit `i` of `matches_mask` is set iff
	 * the last `i + 1` characters of `text` match
	 * the first `i + 1` characters of `pattern` */
	pattern_mask_t matches_mask = 0;
	for (size_t index = 0; index < text_length; index++) {
		unsigned char text_char = text[index];
		/* The last `i + 1` characters match now iff
		 * the last `i` characters matched previously
		 * and `text_char` occurs at index `i + 1` of `pattern`.
		 * Note that the last 0 characters always match, so we shift in a 1 bit.
		 */
		matches_mask = (matches_mask << 1 | 1) & pattern->char_masks[text_char];

		// If bit `pattern_length - 1` of `matches_mask` is set,
		// then the entire pattern matches the preceding characters
		if (matches_mask & pattern->end_mask) {
			// `text[index]` matches the `pattern_length` character of `pattern`,
			// so `text[index - pattern_length + 1]` matches the first character of `pattern`
			match_indices[(*match_count)++] = index - pattern->length + 1;
		}
	}
}
