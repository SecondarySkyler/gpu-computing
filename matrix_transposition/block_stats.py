import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

desktop_block_data = 'csv/desktop_block.csv'
cluster_block_data = 'csv/cluster_block.csv'
desktop_naive_data = 'csv/desktop_naive.csv'
cluster_naive_data = 'csv/cluster_naive.csv'

# load data
desktop_df = pd.read_csv(desktop_block_data, header=None)
cluster_df = pd.read_csv(cluster_block_data, header=None)
desktop_naive_df = pd.read_csv(desktop_naive_data, header=None)
cluster_naive_df = pd.read_csv(cluster_naive_data, header=None)


desktop_df.columns = ['Flag', 'Dimension', 'Times (s)']
cluster_df.columns = ['Flag', 'Dimension', 'Times (s)']
desktop_naive_df.columns = ['Flag', 'Dimension', 'Times (s)']
cluster_naive_df.columns = ['Flag', 'Dimension', 'Times (s)']

# Group by flag and dimension, then calculate the mean time
desktop_mean_times = desktop_df.groupby(['Flag', 'Dimension']).mean().reset_index()
cluster_mean_times = cluster_df.groupby(['Flag', 'Dimension']).mean().reset_index()
desktop_naive_mean_times = desktop_naive_df.groupby(['Flag', 'Dimension']).mean().reset_index()
cluster_naive_mean_times = cluster_naive_df.groupby(['Flag', 'Dimension']).mean().reset_index()

# Calculate bandwidth
for index, row in desktop_mean_times.iterrows():
    total_data = 8 * row['Dimension'] * row['Dimension']
    bandwidth = (total_data / row['Times (s)']) / 1e9
    desktop_mean_times.at[index, 'Bandwidth (GB/s)'] = bandwidth

for index, row in cluster_mean_times.iterrows():
    total_data = 8 * row['Dimension'] * row['Dimension']
    bandwidth = (total_data / row['Times (s)']) / 1e9
    cluster_mean_times.at[index, 'Bandwidth (GB/s)'] = bandwidth

for index, row in desktop_naive_mean_times.iterrows():
    total_data = 8 * row['Dimension'] * row['Dimension']
    bandwidth = (total_data / row['Times (s)']) / 1e9
    desktop_naive_mean_times.at[index, 'Bandwidth (GB/s)'] = bandwidth

for index, row in cluster_naive_mean_times.iterrows():
    total_data = 8 * row['Dimension'] * row['Dimension']
    bandwidth = (total_data / row['Times (s)']) / 1e9
    cluster_naive_mean_times.at[index, 'Bandwidth (GB/s)'] = bandwidth


desktop_naive_mean_times = desktop_naive_mean_times[desktop_naive_mean_times['Dimension'] == 1024]
cluster_naive_mean_times = cluster_naive_mean_times[cluster_naive_mean_times['Dimension'] == 1024]

fig, ax = plt.subplots(1, 2, figsize=(8, 4))
# join the two dataframes on the flag and dimension
result_desktop = pd.merge(desktop_mean_times, desktop_naive_mean_times, on=['Flag', 'Dimension'], suffixes=('_block', '_naive'))
result_cluster = pd.merge(cluster_mean_times, cluster_naive_mean_times, on=['Flag', 'Dimension'], suffixes=('_block', '_naive'))

result_desktop.plot.bar(
    x='Flag', 
    y=['Bandwidth (GB/s)_block', 'Bandwidth (GB/s)_naive'],
    rot=0,
    title='Ryzen 5 5600X (1024x1024)',
    ylabel='Bandwidth (GB/s)',
    ax=ax[0]
)

result_cluster.plot.bar(
    x='Flag', 
    y=['Bandwidth (GB/s)_block', 'Bandwidth (GB/s)_naive'],
    rot=0,
    title='Xeon Silver 4309y (1024x1024)',
    ylabel='Bandwidth (GB/s)',
    ax=ax[1]
)

ax[0].legend(['Block', 'Naive'])
ax[1].legend(['Block', 'Naive'])
# set the layout to tight
plt.tight_layout()

# plt.show()
plt.savefig('./report/img/block_vs_naive.png', dpi=300)