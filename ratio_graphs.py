import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv("data/NEAdV_file.txt")
df2 = pd.read_csv("data/NEAdV_file_w_conc.txt")

plt.rcParams.update({'figure.figsize':(10,5), 
                     'figure.dpi':100})


plt.hist(np.log10(df['Ratio']), 
         bins = 100, 
         histtype = 'step', 
         color = 'mediumseagreen',
         linewidth = 1.2, 
         label='100% PGM returned'
         )

plt.hist(np.log10(df2['Ratio']), 
         bins = 100, 
         histtype = 'step', 
         color = 'dodgerblue',
         linewidth = 1.2, 
         label='184 ppm PGM with flotation'
         )

plt.axvline(x = np.log10(4.18e+4), 
            linestyle = '--', 
            c = 'darkcyan',
            label = 'PGM CO2e/kg'
            )

plt.axvline(x = np.log10(800), 
            linestyle = '--', 
            c = 'dodgerblue',
            label = 'Gold CO2e/kg'
            )

plt.axvline(x = np.log10(1.91), 
            linestyle = '--', 
            c = 'deepskyblue',
            label = 'Iron CO2e/kg'
            )

plt.legend(loc = 'upper right')

plt.gca().set(title='(NEA) Ratio Frequency Histogram', 
              ylabel='Frequency', 
              xlabel = 'Extraterrestrial CO2e/kg returned (log)');

plt.savefig('NEAhistagram')
