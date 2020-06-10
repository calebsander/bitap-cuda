#ifndef BENCH_H
#define BENCH_H

#ifdef BENCH
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

	void start_time(bench_stage_t);
	void stop_time(void);

	void print_bench_times(void);

	#ifdef __cplusplus
	}
	#endif
#else
	#define start_time(stage) do {} while (0)
	#define stop_time() do {} while (0)
#endif

#endif // #ifndef BENCH_H
