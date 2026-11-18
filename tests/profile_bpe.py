from tests.adapters import run_train_bpe
import cProfile
import pstats


profiler = cProfile.Profile()

profiler.runcall(
        run_train_bpe,
        input_path="fixtures/tinystories_sample_5M.txt",
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
)



pstats.Stats(profiler).sort_stats("cumulative").print_stats(15)
