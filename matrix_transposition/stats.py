import pandas as pd
import matplotlib.pyplot as plt

desktop_data = 'csv/desktop_naive.csv'
cluster_data = 'csv/cluster_naive.csv'

#load data
desktop_data_frame = pd.read_csv(desktop_data, header=None)
cluster_data_frame = pd.read_csv(cluster_data, header=None)

desktop_data_frame.columns = ['Flag', 'Dimension', 'Times (s)']
cluster_data_frame.columns = ['Flag', 'Dimension', 'Times (s)']

# Group by flag and dimension, then calculate the mean time
desktop_mean_time = desktop_data_frame.groupby(['Flag', 'Dimension']).mean().reset_index()
cluster_mean_time = cluster_data_frame.groupby(['Flag', 'Dimension']).mean().reset_index()

# Calculate bandwidth
for index, row in desktop_mean_time.iterrows():
    total_data = 8 * row['Dimension'] * row['Dimension']
    bandwidth = (total_data / row['Times (s)']) / 1e9
    desktop_mean_time.at[index, 'Bandwidth (GB/s)'] = bandwidth
    if row['Flag'] == '-O3' and row['Dimension'] == 1024:
        print(f"Desktop: {bandwidth} GB/s")

        

for index, row in cluster_mean_time.iterrows():
    total_data = 8 * row['Dimension'] * row['Dimension']
    bandwidth = (total_data / row['Times (s)']) / 1e9
    cluster_mean_time.at[index, 'Bandwidth (GB/s)'] = bandwidth


# is the same for both dataframes
x_axis = desktop_mean_time.Dimension.unique()

desktop_y_axis_O0 = desktop_mean_time[desktop_mean_time.Flag == '-O0']['Bandwidth (GB/s)']
desktop_y_axis_O1 = desktop_mean_time[desktop_mean_time.Flag == '-O1']['Bandwidth (GB/s)']
desktop_y_axis_O2 = desktop_mean_time[desktop_mean_time.Flag == '-O2']['Bandwidth (GB/s)']
desktop_y_axis_O3 = desktop_mean_time[desktop_mean_time.Flag == '-O3']['Bandwidth (GB/s)']

cluster_y_axis_O0 = cluster_mean_time[cluster_mean_time.Flag == '-O0']['Bandwidth (GB/s)']
cluster_y_axis_O1 = cluster_mean_time[cluster_mean_time.Flag == '-O1']['Bandwidth (GB/s)']
cluster_y_axis_O2 = cluster_mean_time[cluster_mean_time.Flag == '-O2']['Bandwidth (GB/s)']
cluster_y_axis_O3 = cluster_mean_time[cluster_mean_time.Flag == '-O3']['Bandwidth (GB/s)']

# fig, (desktop_plot, cluster_plot) = plt.subplots(2, 1, figsize=(6.4*1.5, 4.8*2))
fig, (desktop_plot, cluster_plot) = plt.subplots(2, 1, figsize=(6.4, 4.8))
fig.tight_layout(pad=3.0)

desktop_plot.plot(x_axis.astype('str'), desktop_y_axis_O0.to_numpy(), label='-O0')
desktop_plot.plot(x_axis.astype('str'), desktop_y_axis_O1.to_numpy(), label='-O1')
desktop_plot.plot(x_axis.astype('str'), desktop_y_axis_O2.to_numpy(), label='-O2')
desktop_plot.plot(x_axis.astype('str'), desktop_y_axis_O3.to_numpy(), label='-O3')
desktop_plot.set_title('AMD Ryzen 5 5600X')

cluster_plot.plot(x_axis.astype('str'), cluster_y_axis_O0.to_numpy(), label='-O0')
cluster_plot.plot(x_axis.astype('str'), cluster_y_axis_O1.to_numpy(), label='-O1')
cluster_plot.plot(x_axis.astype('str'), cluster_y_axis_O2.to_numpy(), label='-O2')
cluster_plot.plot(x_axis.astype('str'), cluster_y_axis_O3.to_numpy(), label='-O3')
cluster_plot.set_title('Intel Xeon Silver 4309y')

desktop_plot.legend()
cluster_plot.legend()

fig.supxlabel('Matrix Dimension')
fig.supylabel('Bandwidth (GB/s)')

# plt.show()
plt.savefig('./report/img/naive_comparison.png', dpi=300)