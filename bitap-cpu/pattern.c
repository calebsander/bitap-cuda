#include "pattern.h"
#include <assert.h>
#include <string.h>

#define MASK_BIT(i) ((pattern_mask_t) 1 << (i))

void preprocess_pattern(const char *pattern, pattern_t *processed_pattern) {
	// Each bit corresponding to a character of `pattern` must fit in a `pattern_mask_t`
	size_t length = strlen(pattern);
	assert(length <= sizeof(pattern_mask_t) * BYTE_BITS);
	processed_pattern->length = length;

	// Construct bitmasks representing where each character appears in `pattern`.
	// Bit `i` of `char_masks[c]` is set iff `c` occurs at index `i + 1` of `pattern`.
	memset(&processed_pattern->char_masks, 0, sizeof(processed_pattern->char_masks));
	for (size_t i = 0; i < length; i++) {
		processed_pattern->char_masks[(unsigned char) pattern[i]] |= MASK_BIT(i);
	}
	if (length) processed_pattern->end_mask = MASK_BIT(length - 1);
}
