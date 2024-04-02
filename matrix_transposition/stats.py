import pandas as pd
import matplotlib.pyplot as plt

data_frame = pd.read_csv(
    'output.csv',
    header=None
)
data_frame.columns = ['Flag', 'Dimension', 'meanBandwidth', 'peakBandwidth']

x_axis = data_frame.Dimension.unique()
y_axis_O0 = data_frame[data_frame.Flag == '-O0']['meanBandwidth']
y_axis_O1 = data_frame[data_frame.Flag == '-O1']['meanBandwidth']
y_axis_O2 = data_frame[data_frame.Flag == '-O2']['meanBandwidth']
y_axis_O3 = data_frame[data_frame.Flag == '-O3']['meanBandwidth']


plt.plot(x_axis.astype('str'), y_axis_O0, label='-O0')
plt.plot(x_axis.astype('str'), y_axis_O1, label='-O1')
plt.plot(x_axis.astype('str'), y_axis_O2, label='-O2')
plt.plot(x_axis.astype('str'), y_axis_O3, label='-O3')
# plt.xticks(x_axis)

plt.legend()
plt.xlabel('Matrix Dimension')
plt.ylabel('Bandwidth (GB/s)')
plt.title('Matrix Transposition')
# plt.show()
plt.savefig('./report/img/output.png')