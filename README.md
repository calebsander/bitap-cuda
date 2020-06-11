# CS 179 Final Project: bitap-cuda

Caleb Sander

06/10/2020

## Summary

I parallelized Wu and Manber's fuzzy bitap algorithm, described in [this paper](https://dl.acm.org/doi/10.1145/135239.135244).
The algorithm is designed to search large bodies of text for approximate matches of a search string, using bitwise operators to test the search string against many offsets in the text simultaneously.
I measured the performance improvements my GPU algorithm obtained over my CPU algorithm.
I was able to achieve a 20x speedup in the core matching algorithm by processing separate blocks of the text simultaneously and by combining the bitmasks of neighboring characters in parallel reductions.
However, the GPU implementation requires some time-consuming additional steps, including copying data between the CPU and GPU, and sorting the matched indices.
Incorporating these additional costs, the performance improvement is closer to 5x.

## Background

The problem of matching strings against each other shows up in many applications.
For example, being able to quickly search within a large set of files is extremely useful when editing a large code repository.
Search engines are solving a version of this problem on a very large scale.
Bioinformatics faces this problem when trying to compare DNA sequences, especially in the presence of genetic mutations.
There are many variants of the string matching problem—looking for exact matches or approximate matches, matching against an entire string or any substring, searching for simple character sequences or complex patterns, etc.

The basic [bitap algorithm](https://en.wikipedia.org/wiki/Bitap_algorithm) implements a search for an exact match of a search string (the "pattern") as a substring of a large chunk of text.
It uses the bit parallelism provided by bitwise operators to simultaneously match every offset of the pattern against each character in the text.
The pattern is first pre-processed to compute an integer `occurrences[c]` for each character value `c`, whose bits indicate the positions where `c` occurs in the pattern.
(For optimal performance, this integer should fit in a machine word, limiting the pattern to 32 or 64 characters.)
Another integer `matches` stores the offsets at which the pattern matches the last few characters of the text, i.e. bit `l` of `matches` indicates whether or not the last `l` characters read from the text match the first `l` characters of the pattern.
When the next character `c` is read from the text, the last `l + 1` characters will match the first `l + 1` characters of the pattern iff the last `l` currently match the pattern and `c` is the character that occurs at position `l + 1` of the pattern.
So `matches` can be updated simply by shifting it left 1 bit and bitwise-anding with `occurrences[c]`.
(Bit 0 of `matches` must also always be set, since there is always a length-0 match.)
Whenever the bit of `matches` corresponding to the length of the pattern is set, the end of an occurrence of the pattern has just been found in the text.

Sun Wu and Udi Manber provide a simple modification of this algorithm that enables it to find *approximate*, or "fuzzy" matches of the pattern that are within a fixed Levenshtein distance of the pattern.
To tolerate up to `D` edits (defined as insertions, deletions, or replacements of a single character), we maintain `D + 1` different `matches` integers: `matches[0]`, ..., `matches[D]`.
Bit `l` in `matches[d]` will be set iff the first `l` characters of the pattern match the preceding characters of text with at most `d` edits.
`matches[0]` is updated just as before, but for `d > 0`, `matches[d]` is updated by accounting for possible edits: bit `l + 1` of `matches[d]` will be set iff any of the following are true:
- bit `l` of the old `matches[d]` is set and the next character also matches
- bit `l` of the old `matches[d - 1]` is set (corresponding to one additional substitution error)
- bit `l + 1` of the old `matches[d - 1]` is set (corresponding to an insertion in the text)
- bit `l` of the new `matches[d - 1]` is set (corresponding to a deletion in the text after this character)

So `matches[d]` can still be computed with a constant number of bitwise operations from the previous values of `matches[d]`, `matches[d - 1]`, and the occurrences of the current character.
(In fact, if the pattern is small enough, we can fit all the bits of the `D + 1` matches masks a single machine word.)
Wu and Manber illustrate many other adaptions of this algorithm that allow more general patterns than string literals (e.g. regular expressions can be matched by constructing an NFA and using the `matches` bits to store which states are currently active).
The error metric can also be adjusted, allowing each insertion, substitution, or deletion to count as a different (integer) number of errors.

## My implementation

To simplify the GPU implementation, I did not implement all of Wu and Manber's extensions to the algorithms.
I focused on the central fuzzy matching algorithm (which I implemented for the CPU and GPU) and also added support for all variants of fixed-length patterns (the CPU and GPU pre-processing is identical).
The CPU implementation directly follows Wu and Manber's paper (the algorithm is at the top of page 86), except I followed the Wikipedia version in inverting the matches mask bits.
By having `0` represent a match and `1` represent no match, the left-shifting operation (which needs new bits to start as a successful match) can use the standard `<<` operator instead of having to shift in a `1` bit.
This modification is especially useful for the GPU reduction, where the shifts are by a variable amount and would otherwise need to shift in the correct number of `1` bits.

I tried several different GPU implementations of the exact matching algorithm to see which would perform best, and they mostly compared as I expected.
The main additional pieces that were needed by all the GPU algorithms were:
- A large buffer to store all indices of the text that match.
	The sequential CPU algorithm can stop at each match and report its index, and then be restarted at the following index.
	However, the GPU algorithm needs to be able to report *all* the matching indices within the chunk of text passed to the kernel.
	It is possible that *every* index of the text matches the pattern (for example, if the pattern is `.` with 0 errors).
	So a very large (uninitialized) buffer needs to be allocated, but this is fairly cheap.
	Appending to the buffer must atomically increment its length, so I used CUDA's `atomicAdd()` for this purpose.
- The returned match indices need to be sorted, so that the matching lines are printed in the correct order and are not printed multiple times if there are multiple matches in the same line.
	I also used the buffer approach in the CPU implementation so that it would have the same interface as the GPU implementation, but it already generates the match indices in sorted order.
	On the GPU, however, the matches can be found in any order by different threads reading different parts of the text, so the matches require explicit sorting.
	Thankfully, this is usually cheap because there are relatively few matches.
	I just used `qsort()` on the CPU, but a GPU sorting kernel would likely be faster if there were many matches.
- Memory needs to be copied to and from the GPU.
	The text makes up the bulk of the memory transfers.
	Some of the others could probably be optimized (e.g. the pattern only needs to be copied to the GPU once even if many chunks of text are being matched), but I didn't attempt this.

The first algorithm I wrote (in `exact_global.cu`) is a simple adaptation of the CPU algorithm: each GPU thread is responsible for finding all matches of the pattern in a small slice of the text.
Within this slice, it reads each character sequentially and uses the normal bitap update algorithm to compute the bitmask after each character.
While this delivered modest improvements over the CPU implementation, it has some significant limitations.
For one, it requires reading overlapping sections of text (`pattern_length - 1` characters need to be read after the end of each slice, in case a match was in progress at the end of the slice).
This provides limiting returns to parallelism, e.g. if the `pattern_length` is 10 and each thread is responsible for a 10-character slice, the entire text is effectively read twice.
Additionally, since neighboring threads are not accessing neighboring characters of the string at the same time, memory accesses are highly uncoalesced.

To address some of these issues without fundamentally changing the algorithm, I wrote a second implementation (in `exact_shared.cu`) where the threads in each thread block cooperatively load their combined slices of text into shared memory.
This ensures that all overlapping slices are only read once (except on thread block boundaries) and also makes it possible to use vectorized loads.
This seemed to slightly improve performance, but required many more thread blocks when processing large chunks of text in order for a single thread block's characters to fit in the 48 KB of shared memory.
(I did discover that it's possible to request several MBs of dynamic shared memory and the kernel will execute very quickly and without error, but return the wrong results.)

I then moved to a parallel reduction algorithm (in `exact_reduce.cu`).
The fine details are described in the comments, but I will give an overview of the algorithm here.
Unlike the previous implementations, each thread reads *one* character at a time; the thread grid runs as many iterations as necessary to cover the text.
Each thread then computes the bitmask corresponding to its character and the threads perform a modified prefix-sum reduction to combine them to compute the complete matches mask after each character in the text.
Notably, this reduction is local, since only the last `pattern_length` characters contribute to each matches mask.
Since the maximum length of the pattern is a compile-time constant (e.g. 32 or 64), I was also able to unroll the up- and down-sweep loops.
The reduction approach substantially improved performance over the two previous algorithms.
(I originally used the 16-byte vectorized loads approach, as in `exact_shared.cu`, but I found it doubled the shared memory usage without improving performance—probably because the reads are already coalesced.)

I further optimized the reduction by using warp shuffles (in `exact_warp_reduce.c`) rather than shared memory to pass values between threads.
Since I was using 32-bit patterns, the local reductions were already being performed on groups of 32 threads, so it was simple to adapt the code to use `__shfl_up_sync()` rather than transfering values through shared memory.
The last matches mask from each warp still needs to be stored in shared memory, since it is used by the following warp as its initial mask.
Despite the reductions in shared memory allocation (32 32-bit integers vs. 1024 32-bit integers), shared memory accesses (converted to shuffles between warp registers instead), and syncpoints, the performance gains were quite small.
I suspect memory access speeds of the text had become the main bottleneck at this point.
In the interest of supporting patterns longer than 32 characters, I switched bace to `exact_reduce.cu` as the default implementation of the GPU exact-matching algorithm.

Using what I had learned from optimizing the exact matching algorithm, I implemented the fuzzy matching algorithm (in `fuzzy_reduce.cu`) based on `exact_reduce.cu`.
The basic approach is the same, but a few modifications were required:
- The parallel reduction needs to be repeated `errors + 1` times, iteratively.
	Notably, only incorporating each thread's character mask into the matches masks requires a reduction.
	The three error cases can be computed from the matches mask at the previous error distance, which will already be stored in shared memory.
	Although it may seem that this inherently sequential part of the algorithm will slow the algorithm down significantly for large error distances, it is probably insignificant because it does not require any additional global memory accesses.
- The mask produced from the error cases needs to be repeatedly combined with the masks produced by the reduction, since it is expected to propagate to subsequent positions in the text

I'm quite pleased with the performance of the algorithm and its similarity to the exact matching algorithm based on parallel reductions.

## Codebase

A large chunk of code, including the pattern preprocessor, the `fgrep`/`agrep` command-line program, and the exact and fuzzy match function interfaces, are shared between the CPU and GPU codebase.
The CPU files (including the ones used by the GPU code) are in the `bitap-cpu` subdirectory.
The files are (the important ones are bolded):
- **`bench.c`**: functions for measuring time intervals and reporting average durations.
	Used by all the benchmarking programs.
- `bench_mmap.c`: benchmark for the CPU fuzzy grep using `mmap()` to load the file
- `bench_read.c`: benchmark for the CPU fuzzy grep using `fread()` to load the file
- **`exact.c`**: the CPU (Wu-Manber) implementation of the exact matcher
- `fuzz_literal_exact.c`: a `libFuzzer` fuzzer used to verify the pattern preprocessor and exact matcher on literal patterns.
	Helped me catch a few instances of undefined behavior.
- **`fgrep.c`**: implements the command-line `fgrep` (exact grepper) and `agrep` (approximate matcher) utilities.
	Used for the GPU implementations as well.
	Currently uses an incremental approach, reading and grepping 128 KB of each file at a time, which is better suited to the CPU matchers.
- **`fuzzy.c`**: the CPU (Wu-Manber) implementation of the fuzzy matcher
- `Makefile`: makes `agrep`, `fgrep`, and the benchmarks for the CPU implementations
- **`pattern.c`**: the pattern preprocessor.
	Can handle character classes and other regular expression shorthand.

A lot of CPU code is guarded by `#ifdef` clauses so it can be reused with minimal duplication.
Here is a complete description of all of the values that can be defined during compilation to enable features:
- `BENCH`: determines whether the benchmarking code is active.
	When enabled, the hooks allow intervals to be timed for benchmark reporting.
	They are disabled for the `agrep`/`fgrep` utilities, and are compiled as no-ops instead.
- `CUDA`: determines whether `fgrep.c` is targeting the GPU matchers.
	Instructs it to allocate page-locked memory for faster transfer to and from the GPU.
- `FUZZY`: determines whether `fgrep.c` compiles `agrep` or `fgrep`.
	If enabled, uses the fuzzy matcher and expects `agrep`'s command line arguments.
- `PATTERN_LENGTH`: set to 32, 64, or 128.
	Determines the maximum length of exact or fuzzy patterns.
	I'm using 32 currently since NVIDIA GPUs are primarily 32-bit machines, but this can be configured in the `Makefile`s.
- `__cplusplus`: all the code is written in C, but CUDA defaults to C++, so we need to add `extern "C"` so that C functions can be called from "C++" code

GPU code files:
- `bench_exact.c`: benchmarks the exact GPU matcher.
	Can be linked with any of the `exact_*.cu` files.
- `bench_fuzzy.c`: benchmarks the fuzzy GPU matcher
- `cuda_utils.h`: defines the `CUDA_CALL()` wrapper to check for errors in CUDA functions, as well as an override so `atomicAdd()` can be used on `size_t` values
- `exact_*.cu`: the four implementations outlined above of exact GPU matchers.
	**`exact_reduce.cu`** is the current default.
- **`fuzzy_reduce.cu`**: the default (and only, currently) implementation of a fuzzy GPU matcher
- `Makefile`: makes `agrep` and `fgrep` for the GPU implementations, as well as the five GPU matching algorithm benchmarks
- `README.md`: you're looking at it

## Executables

You can build the GPU executables by running `make` in the main directory, and the CPU executables by running `make` in the `bitap-cpu` subdirectory.

Either version (CPU or GPU) of `fgrep` can be invoked using `./fgrep pattern [file ...]`.
`pattern` should be the search pattern (e.g. a string literal, or using character class like `.` or `\d`), no more than 32 characters long.
`[file ...]` is an optional list of filenames to search.
If omitted, the standard input is grepped instead.
Matching lines are written to the standard output.

`agrep` can be invoked using `./agrep pattern errors [file ...]`.
The arguments are the same as `fgrep`, except `errors` specifies the maximum number of errors (substitutions, insertions, or deletions) allowed in a match.

The `bench-*` executables don't take any arguments.

I wasn't able to write a test suite, since Titan's GPUs were not responding near the deadline.
But I was planning to just run the CPU and GPU versions of `agrep` on the `oanc.txt` file with various search strings and numbers of errors, and diff their outputs.

## Benchmarking

I made two simple benchmarks to get a sense of how the CPU and GPU algorithms compare.
They grep `oanc.txt`, a 92 MB text file constructed from the [Open American National Corpus](https://www.anc.org/data/oanc/).
Both benchmarks load the entire file into memory before grepping it.
This arguably advantages the GPU implementation since it can benefit from having a large quantity of data to send to the kernel at once.
I also created a version of the CPU fuzzy matching benchmark that uses `mmap()` rather than `fread()` so the file can be read on demand through page-faults, but this does not substantially speed up the CPU algorithm.

The benchmarks distinguish the time spent in each stage of the grep (which are often of very different orders of magnitude).
All stages except `load file` are run 100 times to estimate their mean duration in seconds and the standard deviation of this estimate.
The stages are:
- `load file`: read the file into memory (including allocating memory for the file).
	This stage is only run once, since it's fairly expensive.
- `process pattern`: pattern pre-processing (this is generally negligible, but useful for comparison)
- `allocate indices`: allocates memory for the buffer that will contain the indices of successful matches.
	In principle, this isn't necessary for the CPU matching algorithm, but it was convenient to have the same interface for the CPU and GPU matching functions.
- `copy to gpu`: copy memory from CPU to GPU (only applies for the GPU implementations)
- `find matches`: run the exact/fuzzy matching algorithm.
	This is the main stage of interest.
- `copy from gpu`: copy memory from GPU to CPU (only applies for the GPU implementation)
- `sort matches`: sort the match indices that were found (only applies for the GPU implementation)
- `find lines`: identify the beginning and endings of the lines containing each match index

### Fuzzy: CPU vs. GPU

The fuzzy matching benchmark matches `throughout` against `oanc.txt`, allowing up to 2 errors.
Here are the baseline results for the CPU:

| Reading strategy | Stage | Duration (s) |
| ---------------- | ----- | ------------ |
| `fread()` | load file | 6.018186e-02 |
| `fread()` | process pattern | 7.597200e-07 (+/- 4.046655e-09) |
| `fread()` | allocate indices | 7.448420e-06 (+/- 9.477461e-08) |
| `fread()` | find matches | 3.193866e-01 (+/- 2.665014e-04) |
| `fread()` | find lines | 8.302400e-04 (+/- 1.275973e-06) |
| `fread()` | *TOTAL* | 0.38040690814 |
|
| `mmap()` | load file | 7.374000e-06 |
| `mmap()` | process pattern | 8.175700e-07 (+/- 1.096943e-08) |
| `mmap()` | allocate indices | 8.782880e-06 (+/- 8.969561e-07) |
| `mmap()` | find matches | 3.204796e-01 (+/- 4.433481e-04) |
| `mmap()` | find lines | 8.592020e-04 (+/- 1.265496e-06) |
| `mmap()` | *TOTAL* | 0.32221497845 |

We can see that the `mmap()` implementation is slightly faster because it doesn't have to wait for the entire file to be copied into memory.
But the bulk of the time in both versions is spent in the matching algorithm.

There is only one GPU implementation of the fuzzy matching algorithm to test.
I tried it with several different numbers of thread blocks (each with the maximum 1024 threads) to see the effects of parallelism.
Here is the best-performing one (4096 thread blocks):

| Stage | Duration (s) |
| ----- | ------------ |
| load file | 2.044816e-01 |
| process pattern | 1.362050e-06 (+/- 1.962928e-08) |
| allocate indices | 1.721094e-01 (+/- 5.975975e-05) |
| copy to gpu | 9.257798e-03 (+/- 2.319123e-05) |
| find matches | 1.650310e-02 (+/- 3.425193e-05) |
| copy from gpu | 6.649726e-03 (+/- 9.904235e-06) |
| sort matches | 5.434330e-04 (+/- 1.293656e-06) |
| find lines | 9.049243e-04 (+/- 1.563452e-06) |
| *TOTAL* (without index allocation) | 0.23834194335 |
| *TOTAL* (with index allocation) | 0.41045134335 |

I probably should have modified the index allocation step; on the CPU version, it uses `malloc()`, which is fairly cheap, but on the GPU, it uses `cudaMallocHost()` to obtain a ridiculous 738 MB of page-locked memory.
So I think it should probably be excluded from the total time, as most of the match index buffer is not likely to be used.
Regardless, the file load time now dwarfs the fuzzy match time by an order of magnitude.
It's interesting that the load time is significantly slower; I assume this has to do with the file being loaded into page-locked memory.
The time spent in `find matches` (the actual matching algorithm) is 19.35 times smaller than the CPU implementation.

To get a sense of the impact of thread block count on speed, here's a table showing the number of thread blocks (each with 1024 threads) and the time of just the `find matches` stage, since the rest don't depend on the kernel size:

| Thread blocks | Duration (s) |
| ------------- | ------------ |
| 1 | 4.982508e-01 (+/- 2.717593e-04) |
| 2 | 2.522388e-01 (+/- 3.176426e-04) |
| 4 | 1.266744e-01 (+/- 2.235348e-04) |
| 8 | 6.366972e-02 (+/- 1.114913e-04) |
| 16 | 3.205045e-02 (+/- 6.221669e-05) |
| 32 | 2.328890e-02 (+/- 4.425086e-05) |
| 64 | 2.048365e-02 (+/- 6.818744e-05) |
| 128 | 1.841975e-02 (+/- 8.906235e-05) |
| 256 | 1.749435e-02 (+/- 6.554138e-05) |
| 512 | 1.707598e-02 (+/- 5.176403e-05) |
| 1024 | 1.669651e-02 (+/- 4.543846e-05) |
| 2048 | 1.656482e-02 (+/- 5.022282e-05) |
| 4096 | 1.650310e-02 (+/- 3.425193e-05) |
| 8192 | 1.655064e-02 (+/- 3.446539e-05) |
| 16384 | 1.659383e-02 (+/- 3.156096e-05) |

There are diminishing return to parallelism after 64 thread blocks, with a maximum speed at 4096 blocks, followed by a gradual decline.

### Exact: reduce reuse recycle

To get a sense of which strategy to use to implement the fuzzy GPU algorithm, I benchmarked all four of the exact matching algorithms.
I used the same `oanc.txt` file and grepped for `voyage`.

`exact_global.cu` performed best with 512 thread blocks, each with 1024 threads.
The strategy of dividing the text up into slices, used by `exact_global.cu` and `exact_shared.cu` is well-suited to this benchmark, since it provides such a large block of text to a single kernel invocation.
On more realistic input sizes, they would probably not perform as well.
The breakdown by stage was as follows:

| Stage | Duration (s) |
| ----- | ------------ |
| load file | 2.200820e-01 |
| process pattern | 1.532320e-06 (+/- 2.057660e-08) |
| allocate indices | 1.813877e-01 (+/- 1.607212e-03) |
| copy to gpu | 9.373040e-03 (+/- 3.556898e-05) |
| find matches | 1.702964e-02 (+/- 4.252725e-05) |
| copy from gpu | 6.725547e-03 (+/- 2.139743e-05) |
| sort matches | 3.633900e-06 (+/- 4.839653e-08) |
| find lines | 1.852839e-05 (+/- 8.781201e-08 |
| *TOTAL* (without index allocation) | 0.25323392161 |
| *TOTAL* (with index allocation) | 0.43462162161 |

Note that this isn't an apples-to-apples comparison with the fuzzy matching.
The fuzzy matcher has to compute 3 times more matches masks and also returns fewer matches, so sorting them and copying them from the GPU will be faster.

Next up, `exact_shared.cu`.
I had to use at least 2048 thread blocks to get each block's shared memory to fit in 48 KB.
Here is a list of `find matches` times by different (thread blocks, block size) combinations:

| (Blocks, threads) | Duration (s) |
| ----------------- | ------------ |
| (2048, 32) | 1.779175e-02 (+/- 4.693483e-05) |
| (2048, 64) | 1.279826e-02 (+/- 5.173638e-05) |
| (2048, 128) | 1.290780e-02 (+/- 4.445208e-05) |
| (2048, 256) | 1.303281e-02 (+/- 4.620018e-05) |
| (2048, 512) | 1.320383e-02 (+/- 5.422413e-05) |
| (2048, 1024) | 1.423948e-02 (+/- 4.719191e-05) |
| (4096, 32) | 1.265726e-02 (+/- 4.556824e-05) |
| (4096, 64) | 1.277586e-02 (+/- 5.455628e-05) |
| (4096, 256) | 1.322258e-02 (+/- 5.382679e-05) |
| (8192, 32) | 1.279632e-02 (+/- 4.876413e-05) |
| (8192, 64) | 1.293979e-02 (+/- 5.759963e-05) |

So, a sizeable improvement from 1.702964e-02 down to 1.265726e-02 seconds.

`exact_reduce.cu` brought some additional improvements, especially in requiring many fewer threads.
Here are runtimes `find matches` times for different numbers of 1024-thread blocks:

| Thread blocks | Duration (s) |
| ------------- | ------------ |
| 1 | 1.873079e-01 (+/- 7.399405e-05) |
| 4 | 4.754368e-02 (+/- 1.387421e-05) |
| 16 | 1.246911e-02 (+/- 3.521570e-05) |
| 64 | 1.300440e-02 (+/- 3.013360e-05) |
| 256 | 1.278118e-02 (+/- 4.283544e-05) |
| 1024 | 1.260023e-02 (+/- 3.918499e-05) |
| 4096 | 1.248598e-02 (+/- 4.144924e-05) |

`exact_warp_reduce.cu` provided some small improvements on this, especially across a wider range of kernel sizes.
However, I decided it wasn't worth the reduced flexibility in pattern sizes.

| Thread blocks | Duration (s) |
| ------------- | ------------ |
| 1 | 1.160176e-01 (+/- 1.794003e-04) |
| 4 | 2.943975e-02 (+/- 5.066232e-05) |
| 16 | 1.185328e-02 (+/- 2.402412e-05) |
| 64 | 1.241921e-02 (+/- 5.580896e-05) |
| 256 | 1.247964e-02 (+/- 5.725338e-05) |
| 1024 | 1.248142e-02 (+/- 4.984548e-05) |
| 4096 | 1.263555e-02 (+/- 4.490991e-05) |

I think the main takeaway is that the overhead of accessing such a large amount of memory dominates, regardless of the strategy used to parallelize the algorithm.

Numbers for all the stages of all the benchmarks performed are in [`results.txt`](results.txt).
