#include "fuzzy.h"

void find_fuzzy(
	const pattern_t *pattern,
	size_t errors,
	const char *text,
	size_t text_length,
	size_t *match_indices,
	size_t *match_count
) {
	*match_count = 0;

	/* Compute the initial masks at each error distance.
	 * As in the exact-matching case,
	 * If `distance` errors are allowed, up to `distance` characters of `pattern`
	 * are already matched at the start of `text` due to deletions. */
	pattern_mask_t matches_masks[errors + 1];
	pattern_mask_t initial_mask = matches_masks[0] = 0;
	for (size_t distance = 1; distance <= errors; distance++) {
		initial_mask = matches_masks[distance] = initial_mask << 1 | 1;
	}

	// Accumulate each character of the text into the matches masks
	for (size_t index = 0; index < text_length; index++) {
		// `char_mask` stores which indices of the pattern match this character
		pattern_mask_t char_mask = pattern->char_masks[(unsigned char) text[index]];
		// TODO: newline character should poison surrounding matches.
		// Currently, it can count as an insertion, leading to matches across lines.

		// `last_fewer_errors_mask` stores the matches mask
		// corresponding to 1 fewer errors, before processing this character
		pattern_mask_t last_fewer_errors_mask = matches_masks[0];
		/* `fewer_errors_mask` stores the matches mask
		 * corresponding to 1 fewer errors, after processing this character.
		 * `matches_mask[0]` is updated just as in the exact matcher. */
		pattern_mask_t fewer_errors_mask = matches_masks[0] =
			(last_fewer_errors_mask << 1 | 1) & char_mask;

		for (size_t distance = 1; distance <= errors; distance++) {
			// `last_this_mask` stores this matches mask before processing this character
			pattern_mask_t last_this_mask = matches_masks[distance];
			// There are 4 cases in which the pattern continues to match:
			fewer_errors_mask = matches_masks[distance] =
				// `(last_this_mask << 1 | 1) & char_mask`: no additional errors
				((last_this_mask << 1 | 1) & char_mask) |
				// `last_fewer_errors_mask << 1 | 1`: a substitution at this character
				// `fewer_errors_mask << 1 | 1`: a deletion in the text after this character
				(last_fewer_errors_mask | fewer_errors_mask) << 1 | 1 |
				// `last_fewer_errors_mask`: this character in the text was inserted
				last_fewer_errors_mask;
			last_fewer_errors_mask = last_this_mask;
		}

		/* Check if all the characters in the pattern match with the maximum allowed errors.
		 * We can't tell where the match started since we don't differentiate
		 * insertions vs. deletions vs. substitutions. So just return the end index. */
		if (fewer_errors_mask & pattern->end_mask) {
			match_indices[(*match_count)++] = index + 1;
		}
	}
}
