import pstats

p = pstats.Stats("data/profile/profile_results.prof")
p.strip_dirs().sort_stats("cumulative").print_stats(20)  # Show top 20 functions
