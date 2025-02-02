#ifndef PATTERN_H
#define PATTERN_H

#include <stddef.h>
#include <stdint.h>

#if PATTERN_LENGTH == 32
	typedef uint32_t pattern_mask_t;
#elif PATTERN_LENGTH == 64
	typedef uint64_t pattern_mask_t;
#elif PATTERN_LENGTH == 128
	typedef __uint128_t pattern_mask_t;
#else
	#error "Invalid pattern length"
#endif

#define EOL '\n'
#define BYTE_BITS 8
#define CHAR_VALUES (1 << (sizeof(char) * BYTE_BITS))

typedef struct {
	size_t length;
	pattern_mask_t end_mask;
	pattern_mask_t char_masks[CHAR_VALUES];
} pattern_t;

void preprocess_pattern(const char *pattern, pattern_t *processed_pattern);

#endif // #ifndef PATTERN_H
