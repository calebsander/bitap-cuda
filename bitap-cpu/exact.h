#ifndef EXACT_H
#define EXACT_H

#include "pattern.h"

/**
 * Matches a pattern with no errors in the given text.
 * Returns the starting indices of each match in `match_indices`.
 *
 * @param pattern the pattern being matched
 * @param text the text being matched against (not necessarily null-terminated)
 * @param text_length the length of the text in bytes
 * @param match_indices an array that will store the starting indices of matches.
 *   It must have at least `text_length` entries (in case every index matches).
 * @param match_count will be used to return the number of matches
 */
#ifdef __cplusplus
extern "C"
#endif
void find_exact(
	const pattern_t *pattern,
	const char *text,
	size_t text_length,
	size_t *match_indices,
	size_t *match_count
);

#endif // #ifndef EXACT_H
