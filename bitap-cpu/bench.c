#include "bench.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

/** Human-readable names for the benchmarking stages */
char *STAGE_NAMES[] = {
	"load file",
	"process pattern",
	"allocate indices",
	"copy to gpu",
	"find matches",
	"copy from gpu",
	"sort matches",
	"find lines"
};

/** The time when start_time() was last called */
struct timespec start;
/** The current stage being timed, or BENCH_STAGES if none is being timed */
bench_stage_t bench_stage = BENCH_STAGES;

// The sum of the timed durations (and their squares) for each stage.
// Used to compute the mean and standard deviation of the mean estimate.
double duration_sum[BENCH_STAGES] = {0.0};
double duration_square_sum[BENCH_STAGES] = {0.0};
/** The number of times each stage has been benchmarked */
size_t runs[BENCH_STAGES] = {0};

#define SEC_PER_NS 1e-9

/** Converts the time between two `struct timespec`s into a number of seconds */
double get_duration(struct timespec *start, struct timespec *end) {
	return end->tv_sec - start->tv_sec + (end->tv_nsec - start->tv_nsec) * SEC_PER_NS;
}

void start_time(bench_stage_t stage) {
	assert(bench_stage == BENCH_STAGES); // no timer should be active
	bench_stage = stage;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
}

void stop_time(void) {
	struct timespec end;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end);
	assert(bench_stage != BENCH_STAGES); // some timer should be active
	double duration = get_duration(&start, &end);
	duration_sum[bench_stage] += duration;
	duration_square_sum[bench_stage] += duration * duration;
	runs[bench_stage]++;
	bench_stage = BENCH_STAGES;
}

void print_bench_times(void) {
	assert(bench_stage == BENCH_STAGES); // no timer should still be active
	for (bench_stage_t stage = 0; stage < BENCH_STAGES; stage++) {
		printf("Stage \"%s\": ", STAGE_NAMES[stage]);
		size_t stage_runs = runs[stage];
		if (stage_runs) {
			// Print the mean duration
			double mean = duration_sum[stage] / stage_runs;
			printf("%e s", mean);
			if (stage_runs > 1) {
				// If there are multiple runs to sample from, use the CLT to compute
				// the estimated standard deviation of the mean duration estimate
				double standard_deviation =
					sqrt((duration_square_sum[stage] / stage_runs - mean * mean) / stage_runs);
				printf(" (+/- %e)\n", standard_deviation);
			}
			else putchar('\n');
		}
		else puts("0"); // if stage was never run, assume it wasn't needed
	}
}
