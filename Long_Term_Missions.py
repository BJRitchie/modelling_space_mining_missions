import no_return_funcs as adv
import long_term_funcs as mdv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''
generates the data files for the long-term missions
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


# dV to the moon estimate at a 200km altitude orbit radius
long_term_missions_data = {}
### Lunar data generation --------------------------------------------------------------------------------------
moonDV = mdv.dVtoMoon(ParkingOrbitRadius=rE+200)
lunar_data = mdv.long_term_mission(moonDV,Spcrft_Ve,Spcrft_mtot,mE1,mp1_max,Isp1,mE2,mp2_max,Isp2,vbo,earth_emissions_per_kg, MPE, concentration)
long_term_missions_data['lunar_data'] = lunar_data
print("moon data generated")

### NEA data generation --------------------------------------------------------------------------------------
# nea data has already been generated
nea_data = pd.read_csv('data/NEA_dV_file.txt')
neaDV = nea_data['dV_SH'].median()
nea_data = mdv.long_term_mission(neaDV,Spcrft_Ve,Spcrft_mtot,mE1,mp1_max,Isp1,mE2,mp2_max,Isp2,vbo,earth_emissions_per_kg, MPE, concentration)
long_term_missions_data['nea_data'] = nea_data
print("nea data generated")

### MBA data generation --------------------------------------------------------------------------------------
mba_data = pd.read_csv('data/MBA_dV_file.txt')
mbaDV = mba_data['dV_SH'].median()

mba_data=mdv.long_term_mission(mbaDV, Spcrft_Ve,Spcrft_mtot,mE1,mp1_max,Isp1,mE2,mp2_max,Isp2,vbo,earth_emissions_per_kg, MPE, concentration)
long_term_missions_data['mba_data'] = mba_data
print("mba data generated")


long_term_missions_dataframe = pd.DataFrame.from_dict(long_term_missions_data)
print(long_term_missions_dataframe)

# datacsv = long_term_missions_dataframe.to_csv()

# # creating csv with data for asteroids
# f = open('long_term_missions_data.txt', 'w')
# f.write(datacsv)
# f.close()

earth_emissions_per_kg = 41.8e3         # emissions per kg pgm on earth
MPE = 300/6                             # mass mined per day per kg launched (tonnes/tonnes so no units)
concentration=(184.5)*10**(-6)          # concentration mass of pgm after flotation (%)

destination_list = [['Moon', moonDV], ['NEA', neaDV], ['MBA', mbaDV]]
# fig,ax = plt.subplots(3, figsize=(6,10))
# i=0

# for element in destination_list:
    
#     payload_mass = np.linspace(0, 1e4, 10)
    
#     long_term_mission_data=mdv.long_term_mission(
#         element[1],
#         Spcrft_Ve,
#         payload_mass,
#         mE1,
#         mp1_max,
#         Isp1,
#         mE2,
#         mp2_max,
#         Isp2,
#         vbo,
#         earth_emissions_per_kg,
#         MPE,
#         concentration)

#     mdv.optimum_payload_mass_graph(
#         long_term_mission_data,
#         element[0],
#         earth_emissions_per_kg,
#         MPE,
#         concentration,
#         ax,
#         i,
#     )
#     i = i+1

# plt.tight_layout()
# plt.savefig('NR_missions_BEP_graph')
### LONG-TERM NON-REUSABLE SPACECRAFT -------------------------------------------------------------------

mass_transportation_craft = 3000
extracted_ISPP = adv.extracted_mass()
payload_mass = 6000

i=0
for element in destination_list:
    launches_dict = mdv.single_use_craft(
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

    mdv.single_use_spcraft_graphs(launches_dict, element[0], fignum=i)
    i=i+1

