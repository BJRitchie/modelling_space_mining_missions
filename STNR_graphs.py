import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

'''
graphs for short-term missions with further no returns
'''

df = pd.read_csv("data/NEA_dV_file.txt")
df_conc = pd.read_csv("data/NEAdV_file_w_conc.txt")

plt.rcParams.update({'figure.figsize':(10,5), 
                     'figure.dpi':100})

log_neas = np.log10(df['Ratio'])
log_neas_conc = np.log10(df_conc['Ratio'])

def hist_func(filename):
    plt.hist(log_neas, 
         bins = 100, 
         histtype = 'step', 
         color = 'mediumseagreen',
         linewidth = 1.2, 
         )
    
    plt.hist(log_neas_conc, 
         bins = 100, 
         histtype = 'step', 
         color = 'blue',
         linewidth = 1.2, 
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
                xlabel = 'Extraterrestrial CO2e/kg returned')

    plt.savefig(filename)
        
hist_func('nea_hist')

def proportion_beneficial(df,b):
    
    total = len(df['Ratio'])

    # below = len(df[(df['Ratio'] < 4.18e+4/10**b)])
    below = (df['Ratio'] < 4.18e+4/10**b).sum()
    print(below)

    percent = (below/total) * 100
        
    return below, total, percent

below, total, percent=proportion_beneficial(df, b = 0)

print("NEAs, 100%; {}, {}, {}".format(below, total, percent))

below_conc, total_conc, percent_conc = proportion_beneficial(df_conc, b = 0)

print("NEAs, real conc, {}, {}, {}".format(below_conc, total_conc, percent_conc))


#------- ### MBAs ### ---------------------------------------------------------------------------#

mba_df = pd.read_csv("data/MBA_dV_file.txt")
mba_conc = pd.read_csv("data/MBA_dV_file_conc.txt")

below_mba, total_mba, percent_mba =proportion_beneficial(mba_df, b = 0)
below_mba_conc, total_mba_conc, percent_mba_conc =proportion_beneficial(mba_conc, b = 0)


print("MBAs, 100%; {}, {}, {}".format(below_mba, total_mba, percent_mba))

print("MBAs, real conc, {}, {}, {}".format(below_mba_conc, total_mba_conc, percent_mba_conc))

