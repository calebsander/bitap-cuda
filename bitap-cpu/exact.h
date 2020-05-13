#ifndef EXACT_H
#define EXACT_H

#include "pattern.h"

#define NOT_FOUND SIZE_MAX

size_t find_exact(const pattern_t *pattern, const char *text, size_t text_length);

#endif // #ifndef EXACT_H
