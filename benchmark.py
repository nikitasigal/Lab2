import pandas as pd
import subprocess

sizes = [60, 120, 600, 1200]
proc_counts = [1, 4, 9, 16, 25]
algorithms = ['naive', 'row', 'column', 'block', 'cannon']

df = {algorithm: pd.DataFrame() for algorithm in algorithms}

for algorithm in algorithms:
    times = {proc_count: [] for proc_count in proc_counts}
    for proc_count in proc_counts:
        for size in sizes:
            result = subprocess.run(
                ['mpirun', '-n', f'{proc_count}', '-oversubscribe',
                 './cmake-build-debug/Lab2', f'inputs/input{size}.txt', algorithm],
                stdout=subprocess.PIPE
            ).stdout.decode('utf-8')
            times[proc_count].append(float(result))

    df[algorithm] = pd.DataFrame.from_dict(times, orient='index', columns=sizes)
    df[algorithm].to_csv(f'results/{algorithm}/data.csv')


