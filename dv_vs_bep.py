import MoondVFunctions as mdv
import AsteroiddVFunctions as adv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rE = 6371 # radius of the earth

vbo = 7.58

### Single stage rocket parameters ----------------------------------------------------------------------
Isp = 282                   # Specific impulse
g0 = 9.81*1e-3
ve = g0*Isp                 # Exhaust velocity
mE = (22.2 + 4 + 1.7)*1000  # Total dry mass (kg) (from wiki)
mp_total = 1420000 - mE     # Total propellant mass capacity

### Multi-stage parameters ------------------------------------------------------------------------------
mE1 =  (22.2 +1.7)*1000     # Empty mass of 1st stage (kg) (Including fairing)
mp1_max = 410.9*1000        # Max propellant capacity of 1st stage (kg)
Isp1 = 282                  # Specific impulse of 1st stage
mE2 =  4*1000               # Empty mass of 2nd stage (kg)
mp2_max = 107.5*1000        # Max propellant capacity of 2nd stage (kg)
Isp2 = 311                  # Specific impulse of the second stage

Spcrft_Ve = 444.2*g0        # Ve of the transportation spacecraft
Spcrft_mp = 6000            # 6t of mining equipment
Spcrft_me = 1000            # empty mass of transportation spacecraft
Spcrft_mtot = Spcrft_mp+Spcrft_me


payload_mass = 6000
earth_emissions_per_kg = 41.8e3         # emissions per kg pgm on earth
MPE = 300/6                             # mass mined per day per kg launched (tonnes/tonnes so no units)
concentration=(184.5)*10**(-6)          # concentration mass of pgm after flotation (%)


dataframe = mdv.bep_vs_dV(Spcrft_Ve,payload_mass,mE1,mp1_max,Isp1,mE2,mp2_max,Isp2,vbo, earth_emissions_per_kg,concentration, MPE)

plt.scatter(np.linspace(3,15,13), dataframe['bep_time'])
plt.xlabel('∆V (km/s)')
plt.ylabel('Break Even Point (days)')
plt.savefig('∆V_vs_bep')

