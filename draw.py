import matplotlib.pyplot as plt
import pandas as pd

sizes = [50, 100, 500, 1000]
proc_counts = [1, 4, 9, 16, 25]
algorithms = ['naive', 'row', 'column', 'block', 'cannon']

df = {algorithm: pd.read_csv(f'results/{algorithm}/data.csv', index_col=0) for algorithm in algorithms}

# plot times
for algorithm in algorithms:
    plt.figure(figsize=(10, 5))
    for proc_count in proc_counts:
        plt.plot(range(len(sizes)), df[algorithm].loc[proc_count], label=f'{proc_count} processes')

    plt.xlabel('Size')
    plt.ylabel('Time')
    plt.xticks(range(len(sizes)), sizes)
    plt.legend()
    plt.title('Execution Time')
    plt.savefig(f'results/{algorithm}/time.png')

# plot speedup
for algorithm in algorithms:
    speedup = df[algorithm].loc[1] / df[algorithm]

    plt.figure(figsize=(10, 5))
    for proc_count in proc_counts:
        plt.plot(range(len(sizes)), speedup.loc[proc_count], label=f'{proc_count} processes')

    plt.xlabel('Size')
    plt.ylabel('Speedup')
    plt.xticks(range(len(sizes)), sizes)
    plt.legend()
    plt.title('Speedup')
    plt.savefig(f'results/{algorithm}/speedup.png')

# plot efficiency
for algorithm in algorithms:
    efficiency = (df[algorithm].loc[1] / df[algorithm]).div(df[algorithm].index.to_series(), axis=0)

    plt.figure(figsize=(10, 5))
    for proc_count in proc_counts:
        plt.plot(range(len(sizes)), efficiency.loc[proc_count], label=f'{proc_count} processes')

    plt.xlabel('Size')
    plt.ylabel('Efficiency')
    plt.xticks(range(len(sizes)), sizes)
    plt.legend()
    plt.title('Efficiency')
    plt.savefig(f'results/{algorithm}/efficiency.png')
