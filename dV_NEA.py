import pandas as pd
import no_return_funcs as dv

'''
This script generates a file of data for missions to individual near earth asteroids (NEAs)

it assumes one spacecraft is launched to each asteroid and calculates the emission ratios for each asteroid

emission ratios are based on the fuel emitted in the atmosphere upon earth launch and the mass that can be brought back from the asteroid

'''

### Generating dataframe -----------------------------------------------------------------------------

df = pd.read_csv("data/NEA_Asteroids.csv")

df = dv.dV_SH(df)

### Inputs --------------------------------------------------------------------------------------------

Spcrft_Me = 3000                        # empty spacecraft mass
Spcrft_Ve = 0.0098*444.2                # average of engines on wiki

# Fuel Required to Leave Atmosphere ---------------------------------------------------------------------------
# Delta-V to 200 km orbit
vbo = 7.58 #

# Single stage rocket parameters
Isp = 282.   # Specific impulse
g0 = 9.81*1e-3
ve = g0*Isp # Exhaust velocity
mE = (22.2 + 4 + 1.7)*1000 # Total dry mass (kg) (from wiki)
mp_total = 1420000 - mE    # Total propellant mass capacity
mex = dv.extracted_mass()

# Multi-stage parameters
mE1 =  (22.2 +1.7)*1000 # Empty mass of 1st stage (kg) (Including fairing)
mp1_max = 410.9*1000   # Max propellant capacity of 1st stage (kg)
Isp1 = 282             # Specific impulse of 1st stage
mE2 =  4*1000           # Empty mass of 2nd stage (kg)
mp2_max = 107.5*1000    # Max propellant capacity of 2nd stage (kg)
Isp2 = 311 # Specific impulse of the second stage

# Emission Indexes --------------------------------------------------------------------------------------------
EICO2 = 0.65
EIH2O = 0.35
EIBC = 0.02
EINO2 = 0.02

Ratio = dv.emissions(df, 
                    EICO2 = 0.65, 
                    EIH2O = 0.3, 
                    EIBC = 0.02, 
                    EINO2 = 0.02, 
                    mE1 =  (22.2 +1.7)*1000,
                    mp1_max = 410.9*1000,
                    Isp1 = 282 ,
                    mE2 =  4*1000,
                    mp2_max = 107.5*1000 ,
                    Isp2 = 311,
                    vbo = 7.58, 
                    Spcrft_Me = 4000, 
                    Spcrft_Ve = 0.0098*444.2, 
                    mex = dv.extracted_mass() 
                    )

print('test_ratio: Ratio calculated')

print("mean of stage1: "+str(df['mPropStage1'].mean()))
print("median of stage1: "+str(df['mPropStage1'].median()))

print("mean of stage2: "+str(df['mPropStage2'].mean()))
print("median of stage2: "+str(df['mPropStage2'].median()))

dfcsv = df.to_csv()

# creating csv with data for asteroids
f = open('data/NEAdV_file_w_conc.txt', 'w')
f.write(dfcsv)
f.close()

