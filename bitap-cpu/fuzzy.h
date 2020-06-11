#ifndef FUZZY_H
#define FUZZY_H

#include "pattern.h"

/**
 * Matches a pattern with some errors in the given text.
 * Returns the *ending* indices of each match in `match_indices`.
 *
 * @param pattern the pattern being matched
 * @param errors the number of errors (insertions, deletions, or substitutions) allowed
 * @param text the text being matched against (not necessarily null-terminated)
 * @param text_length the length of the text in bytes
 * @param match_indices an array that will store the starting indices of matches.
 *   It must have at least `text_length` entries (in case every index matches).
 * @param match_count will be used to return the number of matches
 */
void find_fuzzy(
	const pattern_t *pattern,
	uint32_t errors,
	const char *text,
	size_t text_length,
	size_t *match_indices,
	size_t *match_count
);

#endif // #ifndef FUZZY_H
