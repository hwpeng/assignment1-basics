import time

from adapters import run_train_bpe

start = time.perf_counter()

vocab, merges = run_train_bpe(
    input_path="fixtures/tinystories_sample_5M.txt",
    vocab_size=10_000,
    special_tokens=["<|endoftext|>"],
)

elapsed = time.perf_counter() - start
print(f"Training took {elapsed:.2f} seconds")
