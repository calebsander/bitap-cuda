#ifndef FUZZY_H
#define FUZZY_H

#include "pattern.h"

#define NOT_FOUND SIZE_MAX

size_t find_fuzzy(
	const pattern_t *pattern,
	size_t errors,
	const char *text,
	size_t text_length
);

#endif // #ifndef FUZZY_H
