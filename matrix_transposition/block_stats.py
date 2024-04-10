import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

m1_block_data = 'm1_block.csv'
m1_data = 'output_m1.csv'

m1_block_data_frame = pd.read_csv(
    m1_block_data,
    header=None
)

m1_data_frame = pd.read_csv(
    m1_data,
    header=None
)

m1_block_data_frame.columns = ['Flag', 'Dimension', 'Bandwidth']
m1_data_frame.columns = ['Flag', 'Dimension', 'Bandwidth', 'peakBandwidth']
x = np.arange(len(m1_block_data_frame['Flag']))

# select only the rows where dimension is 1024
m1_data_frame = m1_data_frame[m1_data_frame.Dimension == 1024]

barWidth = 0.25
bars_O0 = []
bars_O1 = []

for i in range(4):
    bars_O0.append(m1_data_frame.iloc[i]['Bandwidth'])
    bars_O1.append(m1_block_data_frame.iloc[i]['Bandwidth'])


r1 = np.arange(len(bars_O0))
r2 = [x + barWidth for x in r1]

plt.bar(r1, bars_O0, width=barWidth, edgecolor='grey', label='Naive')
plt.bar(r2, bars_O1, width=barWidth, edgecolor='grey', label='Block')

plt.xlabel('Optimization Levels', fontweight='bold')
plt.ylabel('Bandwidth (GB/s)', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(bars_O0))], ['-O0', '-O1', '-O2', '-O3'])

plt.legend()
plt.title('Naive vs Block Transposition (1024x1024, M1 chip, block size = 32)')
plt.show()


