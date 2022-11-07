import no_return_funcs as nrf
import long_term_funcs as ltf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''
this generates the break even point graphs for:
- NEAs
- MBAs
- moon

assuming:
- a long term mining base
- a single-use transportation spacecraft
'''

### VARIABLES -------------------------------------------------------------------------------------------
rE = 6371 # radius of the earth (km)

# burnout velocity for a 200km parking orbit:
vbo = 7.58

### Single stage rocket parameters ----------------------------------------------------------------------
Isp = 282                   # Specific impulse                  (s)
g0 = 9.81*1e-3              # acceleration due to gravity       (km/s2)
ve = g0*Isp                 # Exhaust velocity                  (km/s)
mE = (22.2 + 4 + 1.7)*1000  # Total dry mass                    (kg) (from wiki)
mp_total = 1420000 - mE     # Total propellant mass capacity    (kg)

### Multi-stage parameters ------------------------------------------------------------------------------
mE1 =  (22.2 +1.7)*1000     # Empty mass of 1st stage                   (kg) (Including fairing)
mp1_max = 410.9*1000        # Max propellant capacity of 1st stage      (kg)
Isp1 = 282                  # Specific impulse of 1st stage             (s)
mE2 =  4*1000               # Empty mass of 2nd stage                   (kg)
mp2_max = 107.5*1000        # Max propellant capacity of 2nd stage      (kg)
Isp2 = 311                  # Specific impulse of the second stage      (s)

Spcrft_Ve = 444.2*g0        # Ve of the transportation spacecraft       (km/s)
Spcrft_mp = 5000            # Mass of payload (mining equipment)        (kg)                
Spcrft_me = 1000            # empty mass of transportation spacecraft   (kg)
Spcrft_mtot = Spcrft_mp+Spcrft_me

earth_emissions_per_kg = 41.8e3         # emissions per kg pgm on earth
MPE = 300/6                             # mass mined per day per kg launched (tonnes/tonnes so no units)
concentration=(184.5)*10**(-6)          # concentration mass of pgm after flotation (%)

mba_data = pd.read_csv('data/MBA_dV_file_conc.txt')
nea_data = pd.read_csv('data/NEA_dV_file.txt')

moonDV = ltf.dVtoMoon(ParkingOrbitRadius=rE+200)
neaDV = nea_data['dV_SH'].median()
mbaDV = mba_data['dV_SH'].median()

mass_transportation_craft = 3000
extracted_ISPP = nrf.extracted_mass()
payload_mass = 6000

destination_list = [['Moon', moonDV], ['NEA', neaDV], ['MBA', mbaDV]]

fig,ax = plt.subplots(3, sharey=True, figsize=(6,10))
i=0

for element in destination_list:
    launches_dict = ltf.single_use_craft(
        element[1],
        Spcrft_Ve,
        mass_transportation_craft,
        extracted_ISPP,
        earth_emissions_per_kg,
        MPE,
        concentration,
        payload_mass,
        mE1,
        mp1_max,
        Isp1,
        mE2,
        mp2_max,
        Isp2,
        vbo
    )

    ltf.SUS_BEP_graph(
        launches_dict, 
        figname = element[0], 
        earth_emissions_per_kg=earth_emissions_per_kg, 
        concentration=concentration*50,  # concentration times the increase from flotation
        ax=ax, 
        fignum=i)
    i=i+1

plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('NR_missions_BEP_graph')

