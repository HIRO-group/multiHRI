import pstats
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("name", help="name of the profile file")
args = parser.parse_args()
name = args.name
p = pstats.Stats(f"data/profile/{name}")
p.strip_dirs().sort_stats("cumulative").print_stats(20)  # Show top 20 functions
