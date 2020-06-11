#ifndef BENCH_H
#define BENCH_H

#ifdef BENCH
	/** The stages of a CPU/GPU grep */
	typedef enum {
		LOAD_FILE,
		PROCESS_PATTERN,
		ALLOCATE_INDICES,
		COPY_TO_GPU,
		FIND_MATCHES,
		COPY_FROM_GPU,
		SORT_MATCHES,
		FIND_LINES,

		BENCH_STAGES
	} bench_stage_t;

	#ifdef __cplusplus
	extern "C" {
	#endif

	/** Starts timing the given stage (only one timer can be active at once) */
	void start_time(bench_stage_t);
	/** Stops timing the current stage and saves its duration */
	void stop_time(void);

	/** Prints the mean and standard deviation of all stage time estimates */
	void print_bench_times(void);

	#ifdef __cplusplus
	}
	#endif
#else
	// These "functions" are no-ops when benchmarking is disabled
	#define start_time(stage) do {} while (0)
	#define stop_time() do {} while (0)
#endif

#endif // #ifndef BENCH_H
