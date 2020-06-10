#include "bench.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

char *STAGE_NAMES[] = {
	"load file",
	"process pattern",
	"allocate indices",
	"copy to gpu",
	"find matches",
	"copy from cpu",
	"sort matches",
	"find lines"
};

struct timespec start;
bench_stage_t bench_stage = BENCH_STAGES;
double duration_sum[BENCH_STAGES] = {0.0};
double duration_square_sum[BENCH_STAGES] = {0.0};
size_t runs[BENCH_STAGES] = {0};

#define SEC_PER_NS 1e-9

double get_duration(struct timespec *start, struct timespec *end) {
	return end->tv_sec - start->tv_sec + (end->tv_nsec - start->tv_nsec) * SEC_PER_NS;
}

void start_time(bench_stage_t stage) {
	assert(bench_stage == BENCH_STAGES);
	bench_stage = stage;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
}

void stop_time(void) {
	struct timespec end;
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end);
	assert(bench_stage != BENCH_STAGES);
	double duration = get_duration(&start, &end);
	duration_sum[bench_stage] += duration;
	duration_square_sum[bench_stage] += duration * duration;
	runs[bench_stage]++;
	bench_stage = BENCH_STAGES;
}

void print_bench_times(void) {
	assert(bench_stage == BENCH_STAGES);
	for (bench_stage_t stage = 0; stage < BENCH_STAGES; stage++) {
		printf("Stage \"%s\": ", STAGE_NAMES[stage]);
		size_t stage_runs = runs[stage];
		if (stage_runs) {
			double mean = duration_sum[stage] / stage_runs;
			printf("%e s", mean);
			if (stage_runs > 1) {
				double standard_deviation =
					sqrt((duration_square_sum[stage] / stage_runs - mean * mean) / stage_runs);
				printf(" (+/- %e)\n", standard_deviation);
			}
			else putchar('\n');
		}
		else puts("0");
	}
}
