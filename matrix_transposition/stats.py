import pandas as pd
import matplotlib.pyplot as plt

m1_data = 'csv/output_m1.csv'
desktop_data = 'csv/desktop_naive.csv'
cluster_data = 'csv/cluster_naive.csv'

#load data
m1_data_frame = pd.read_csv(m1_data, header=None)
desktop_data_frame = pd.read_csv(desktop_data, header=None)
cluster_data_frame = pd.read_csv(cluster_data, header=None)

m1_data_frame.columns = ['Flag', 'Dimension', 'meanBandwidth', 'peakBandwidth']
desktop_data_frame.columns = ['Flag', 'Dimension', 'meanBandwidth']
cluster_data_frame.columns = ['Flag', 'Dimension', 'meanBandwidth']

# is the same for both dataframes
x_axis = m1_data_frame.Dimension.unique()

m1_y_axis_O0 = m1_data_frame[m1_data_frame.Flag == '-O0']['meanBandwidth']
m1_y_axis_O1 = m1_data_frame[m1_data_frame.Flag == '-O1']['meanBandwidth']
m1_y_axis_O2 = m1_data_frame[m1_data_frame.Flag == '-O2']['meanBandwidth']
m1_y_axis_O3 = m1_data_frame[m1_data_frame.Flag == '-O3']['meanBandwidth']

desktop_y_axis_O0 = desktop_data_frame[desktop_data_frame.Flag == '-O0']['meanBandwidth']
desktop_y_axis_O1 = desktop_data_frame[desktop_data_frame.Flag == '-O1']['meanBandwidth']
desktop_y_axis_O2 = desktop_data_frame[desktop_data_frame.Flag == '-O2']['meanBandwidth']
desktop_y_axis_O3 = desktop_data_frame[desktop_data_frame.Flag == '-O3']['meanBandwidth']

cluster_y_axis_O0 = cluster_data_frame[cluster_data_frame.Flag == '-O0']['meanBandwidth']
cluster_y_axis_O1 = cluster_data_frame[cluster_data_frame.Flag == '-O1']['meanBandwidth']
cluster_y_axis_O2 = cluster_data_frame[cluster_data_frame.Flag == '-O2']['meanBandwidth']
cluster_y_axis_O3 = cluster_data_frame[cluster_data_frame.Flag == '-O3']['meanBandwidth']

fig, (m1_plot, desktop_plot, cluster_plot) = plt.subplots(3, 1, figsize=(6.4*1.5, 4.8*2))
fig.tight_layout(pad=3.0)

m1_plot.plot(x_axis.astype('str'), m1_y_axis_O0.to_numpy(), label='-O0')
m1_plot.plot(x_axis.astype('str'), m1_y_axis_O1.to_numpy(), label='-O1')
m1_plot.plot(x_axis.astype('str'), m1_y_axis_O2.to_numpy(), label='-O2')
m1_plot.plot(x_axis.astype('str'), m1_y_axis_O3.to_numpy(), label='-O3')
m1_plot.set_title('Apple M1')

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

m1_plot.legend()
desktop_plot.legend()
cluster_plot.legend()
flags = ['-O0', '-O1', '-O2', '-O3']

fig.supxlabel('Matrix Dimension')
fig.supylabel('Bandwidth (GB/s)')

# plt.show()
plt.savefig('./report/img/naive_comparison_2.png', dpi=300)