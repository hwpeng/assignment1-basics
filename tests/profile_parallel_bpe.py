import cProfile
import os
import pstats
import time
import tests.bpe_parallel as parallel

original_count_chunk = parallel.count_chunk

def profiled_chunk(args):
    profiler = cProfile.Profile()
    result = profiler.runcall(original_count_chunk, args)
    profiler.dump_stats(f'/tmp/cs336-bpe-worker-{os.getpid()}-{args[1]}.prof')
    return result

def main():
    from tests.adapters import run_train_bpe
    original_parallel = parallel.parallel_pre_token_counts
    def timed_parallel(*args, **kwargs):
        start = time.perf_counter()
        result = original_parallel(*args, **kwargs)
        print(f'Parallel pre-tokenization wall time (with worker profiling): {time.perf_counter()-start:.2f}s', flush=True)
        return result
    parallel.count_chunk = profiled_chunk
    parallel.parallel_pre_token_counts = timed_parallel
    profiler = cProfile.Profile()
    start = time.perf_counter()
    profiler.runcall(run_train_bpe, 'data/TinyStoriesV2-GPT4-train.txt', 10000, ['<|endoftext|>'], strategy='heap', num_processes=8)
    print(f'Full run wall time (with profiling): {time.perf_counter()-start:.2f}s', flush=True)
    profiler.dump_stats('/tmp/cs336-bpe-parent.prof')
    pstats.Stats(profiler).strip_dirs().sort_stats('cumulative').print_stats(18)

if __name__ == '__main__':
    main()
