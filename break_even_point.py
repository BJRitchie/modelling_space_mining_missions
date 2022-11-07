import long_term_funcs as ltf
import Long_Term_Missions as ltm  # for variables
import pandas as pd 

'''
generates break even point for a long-term mission to the moon
'''

data = pd.read_csv('data/long_term_missions_data.txt')

earth_emissions_per_kg = 41.8e3 # emissions per kg pgm returned
MPE = 50  # mass mined per day per kg launched
concentration = (184.5)*10**(-6)

# graph of break even point in time and launch payload mass
ltf.optimum_payload_mass(
    MPE=MPE,
    earth_emissions_per_kgram=earth_emissions_per_kg,
    dv=ltm.moonDV,
    Spcrft_Ve=ltm.Spcrft_Ve,
    mE1=ltm.mE1,
    mp1_max=ltm.mp1_max,
    Isp1=ltm.Isp1,
    mE2=ltm.mE2,
    mp2_max=ltm.mp2_max,
    Isp2=ltm.Isp2,
    vbo=ltm.vbo,
    concentration=concentration,
    filename = 'lunar_LTM'
    )

### can toggle for testing 

# bep_mass_wo_BC, bep_time_wo_BC = mdv.break_even_point(data['mba_data'][8]-data['mba_data'][6], earth_emissions_per_gram, mining_rate)
# bep_mass, bep_time = mdv.break_even_point(data['mba_data'][8]-data['mba_data'][6], earth_emissions_per_gram, mining_rate)

# print("the BE mass is "+str(bep_mass_wo_BC))
# print("the BE time (days) is "+str(bep_mass_wo_BC))

# years = bep_time_wo_BC // 365 
# days = bep_time_wo_BC % 365 
# print("it takes {} years and {} days to break even".format(years,days)) 

# kg_pgms = [(300e3)*(7.725e-4), (300e3)*(1.545e-3)] 
# print(kg_pgms) 


