# import matplotlib.pyplot as plt
import numpy as np

'''
this file defines all the functions for the long-term mission modeling

no_return functions are optimised for lists of variables, whereas this mission 
architecture will only use a single variable for dV
- resulted in some odd syntax as the no_return funcs were written well before these mission architectures were investigated

- Median dVs for NEAs and MBAs
- An estimate for the dV to the moon
'''

def dVtoMoon(ParkingOrbitRadius):
    '''
    based on process outlined in ch9 of "Orbital Mechanics for Engineering Students, H.D. Curtis

    This is a simple model to obtain an estimate of the dV to the moon

    - it estimates based on a Hohmann transfer going from a circular parking orbit to an 
    elliptical transfer orbit to the moon, followed by a burn to enter lunar orbit
    - ∆V is estimated by comparing orbit velocities

    '''
    
    ### CONSTANTS ------------------------------------------------------------------------------------
    LUNAR_ORBIT = 384000        # km (radius assumed circular)
    UE = 398600                 # KM**3/S**2 (earths grav parameter)
    VMOON = np.sqrt(UE/LUNAR_ORBIT)

    ### VARIABLES ------------------------------------------------------------------------------------
    # velocity in parking orbit
    vc = np.sqrt(UE/ParkingOrbitRadius)
    #  angular momentum of transfer ellipse
    h = np.sqrt(2*UE)*np.sqrt((ParkingOrbitRadius*LUNAR_ORBIT)/(ParkingOrbitRadius+LUNAR_ORBIT))
    # speeds vp and va, at perigee and apogee respectively
    vp = h/ParkingOrbitRadius
    va = h/LUNAR_ORBIT
    # dvp to enter transfer ellipse from circular orbit
    dvp = vp - vc
    # dva required to transfer from hohmann ellipse to the moon's orbit
    dva = VMOON - va
    # dvtot total dv of entire transfer
    dvtot = dvp + dva

    return dvtot

def lunar_payload_mass(dv, Spcrft_Me, Spcrft_Ve):
    '''
    Payload mass upon launch ------------------------------------------------------------------------
    This mass is the only variable that changes for the launch and is dependent upon the dV required 
    to reach the desired asteroid
    
    '''
    
    Spcrft_mp = Spcrft_Me*(np.exp(dv/Spcrft_Ve)-1)
    mpl =  Spcrft_mp + Spcrft_Me
        
    return mpl

def long_term_mission_emissions(mpropellant):
    
    #### Variables - from "Radiative_Forcing_Caused_by_Rocket_Engine_Emission.pdf" MN Ross
    # ------------------------------------------------------------------------------------------------------------ #

    ### GWP values - CO2e per kg component
    GWP_CO2_ = 1
    GWP_BC_ = 50000
    GWP_NO2_ = 281.5
    
    ### emission indexes (percentage by mass of the exaust given by co2, h2o, black carbon and no2) ----------------
    # ------------------------------------------------------------------------------------------------------------ #
    EICO2 = 0.65
    EIH2O = 0.35
    EIBC = 0.02
    EINO2 = 0.02
    
    ### kgs of each component emitted
    # multiplied by 2/3 because this is the average amount of fuel burnt above the troposphere in a rocket launch 
    # (Ross's paper^)

    CO2 = EICO2 * mpropellant

    H2O = EIH2O * (2/3) * mpropellant

    BC = EIBC * (2/3) * mpropellant
    
    NO2 = EINO2 * mpropellant

    ### Global Warming Potential for each component
    GWP_BC = GWP_BC_ * BC

    GWP_NO2 = GWP_NO2_ * NO2

    GWP_tot = np.array(CO2) + np.array(GWP_BC) + np.array(GWP_NO2)
    
    return CO2, H2O, BC, NO2, GWP_BC, GWP_NO2, GWP_tot

def moon_prop_mass_two_stage(mE1,mp1_max,Isp1,mE2,mp2_max,Isp2,mpl,vbo):
    ''' 
    Compute the propellant mass for a two-stage launch vehicle to reach a
    specified burnout velocity (of delta-V).
    
    This function solves an optimization problem to minimize the total launch
    mass of the launch vehicle, with constraints on the propellant capacity of
    each stage, and the burnout velocity (delta-V) achieved. 
    
    Warning: This function uses an iterative method to solve for each payload 
    mass mpl[i] - hence it may be slow when computing for a large number of values.
    For many values, use the prop_mass_two_stage_polyfit() function that uses a 
    polynomial fit to the mpl vs mp1, mpl vs mp2 curves.

    Parameters
    ----------
    mE1 : TYPE
        Empty mass of 1st stage (kg)
    mp1_max : TYPE
        Maximum propellant mass of 1st stage (kg)
    Isp1 : TYPE
        Specific impulse of 1st stage (s)
    mE2 : TYPE
        Empty mass of 2nd stage (kg)
    mp2_max : TYPE
        Maximum propellant mass of 2nd stage (kg)
    Isp2 : TYPE
        Specific impulse of 2nd stage (s)
    mpl : TYPE
        Payload mass (kg)
    vbo : TYPE
        Burnout velocity (km/s)

    Returns
    -------
    mp : TYPE
        DESCRIPTION.
    
    e1,e2,n, # Structure ratios
    mp1, mp2 # Propellant masses of each stage
    coefs1a,coefs2a,coefs1b,coefs2b # Coefficients of the polynomial fits
    mpl_critical # Critical payload mass separating the two trend lines
    
    '''
    
    from scipy import optimize
    # import numpy.polynomial.polynomial as poly
    
    
    # Format inputs
    # if type(mpl)==float:
    #     # Only single payload mass requested.
    #     # Turn this into an array to generalize method.
    #     mpl = np.array([mpl])

    
    
    # Compute exhaust velocities (km/s)
    g0 = 9.81*1e-3 
    c1 = Isp1*g0 # 1st stage
    c2 = Isp2*g0 # 2nd stage
    
    # Optimal staging
    # Each stage i has an empty mass mEi, propellant mass mpi, and total mass mi.
    # For convenience, define the structural ratio εi = mEi/mi and the payload
    # fraction λi = mpli/(mEi + mpi)
    # Total mass is m0 = sum(mi)
    # Total burnout velocity is vbo = sum(ci*ln(ni) ),
    # where ni = n = (1 + λi)/(εi + λi)
    
    
    # Method 1: Lagrange Multipliers ------------------------------------------
    
    # Use Lagrange multipliers method to find the optimal mass of each stage
    # to achieve a required burn out velocity, with minimum total mass m0.
    #
    # Following method in Curtis (2005) S11.6.1 (pg 571-578)
    # Solve a system of non-linear equations, in terms of the unknown structural
    # ratios ε1, ε2, and the Lagrange multiplier η.
    
    def equations(p):
        ''' System of non-linear equations to solve '''
        e1,e2,n = p
        
        f1 = (c1*n - 1)/(c1*e1*n) - (mE1/e1 + mE2/e2 + mpl)/(mE1 + mE2/e2 + mpl)
        f2 = (c2*n - 1)/(c2*e2*n) - (mE2/e2 + mpl)/(mE2 + mpl)
        #f3 = c1*np.log((c1*n - 1)/(c1*e1*n)) + c2*np.log((c2*n - 1)/(c2*e2*n)) - vbo
        f3 = c1*np.log( (mE1/e1 + mE2/e2 + mpl)/(mE1 + mE2/e2 + mpl) ) + c2*np.log( (mE2/e2 + mpl)/(mE2 + mpl) ) - vbo
        
        return (f1,f2,f3)
    
    # Method 2: Constrained Optimization --------------------------------------
    
    # Objective function to minimize (total mass)
    def m0(p):
        e1,e2 = p
        return mE1/e1 + mE2/e2 + mpl
    
    def constraint(p):
        ''' 
        Burnout velocity constraint
        g(x) = required burnout - achieved burnout
        for success, need g(x) < 0
        '''
        e1,e2 = p
        g = vbo - (c1*np.log( (mE1/e1 + mE2/e2 + mpl)/(mE1 + mE2/e2 + mpl) ) + c2*np.log( (mE2/e2 + mpl)/(mE2 + mpl) ))
        
        return g
    
    # get machine precision for bounds
    eps = np.finfo(float).eps
    
    # Find minimum values of e1,e2 from propellant mass
    e1_min = 1/(mp1_max/mE1 + 1)
    e2_min = 1/(mp2_max/mE2 + 1)
    
    
    # Solve the equations -----------------------------------------------------

    # Vector of payload masses
    mpl_copy = mpl.copy() # Copy original vector
    e1 = np.zeros(len(mpl))*np.nan # Initialize vector of e1 results 
    e2 = np.zeros(len(mpl))*np.nan # Initialize vector of e2 results 
    n = np.zeros(len(mpl))*np.nan  # Initialize vector of n results
    
    # Preliminaries.
    # Check if requested payload can be delivered by th upper stage alone.
    import no_return_funcs as adv
    mp2_vec = adv.prop_mass_single_stage(mE2,mpl,Isp2,vbo) # Propellant masses
    upper_stage_only = mp2_vec < mp2_max
    # Upper_stage_only flag:
    # True : payload can be launched with just the upper stage.
    #        Set 1st stage as empty e1 = 0; e2 = 1 as initial guess
    # False: need both stages to reach orbit.
    #        set 
    
    # Loop through
    
    print('prop_mass_two_stage: solving equations')

    import math
    for j, mpl in enumerate(mpl_copy): # Iterate through
        
        
        # Method 1:
        e1j,e2j,nj = optimize.fsolve(equations,(0.1,0.1,10.))
        
        
        # Method 2: Minimize m0 subject to bounds and constraint
        # pdb.set_trace()
        
        # Initial guess x0
        margin = 1e-5 # Add margin to the bounds
            
        if upper_stage_only[j] == True:
            # Use upper stage only.
            # Set e1 = 1 (empty first stage)
            #     e2 = e2_min
            x0 = np.array([1.,e2_min])
        else:
            # Need both stages
            x0 = np.array([e1_min,e2_min])
        
        # Solve the equation
        try:
            # Try to solve the equation
            soln = optimize.minimize(m0,  # Objective function to minimize
                                     x0,  # Inital guess (e1,e2)
                                     # bounds=((0, 1.), (0., 1.)), # Perfect bounds
                                     bounds=((e1_min, 1.-margin), (e2_min, 1.-margin)), # Realistic bounds (force remove e1=0,1)
                                     constraints={"fun": constraint, "type": "eq",     # Burnout constraint
                                                  # "fun": lambda x: mE1/x[0] - mE1 -mp1_max, "type":'ineq', #  Max prop constraint 1st stage mp1 < mp1_max
                                                  # "fun": lambda x: mE2/x[1] - mE2 -mp2_max, "type":'ineq', #  Max prop constraint 2nd stage mp1 < mp1_max
                                                  },
                                      # method="SLSQP",
                                      )
        except:
            # No solution found. Set values at nan and continue.
            e1[j] = np.nan
            e2[j] = np.nan
            continue
            
        # Extract solution to the equation
        e1j, e2j = soln.x
        
        # Check vbo constrain
        g = constraint(soln.x)
        if (g > 0):
            
            # Constraint not met. 
            # Check precision
            if math.isclose(0, g, abs_tol=1e-5) == False:
                # G is much greater than zero.
                # Constraint is not met
                e1[j] = np.nan
                e2[j] = np.nan
                continue # Skip to next iteration
            else:
                # G is only slightly positive.
                # Within accepted values. Accept solution
                pass
        
        # Save results
        e1[j] = e1j
        e2[j] = e2j
    
    
    # Curve smoothing ---------------------------------------------------------
    # Split the results into two domains
    # a. Upper stage only
    # b. Both stages
    # In each domain, fit a polynomial to the data. Use this polynomial to
    # replace any missing values.
    
    
    # a. Upper stage only
    # Extract data
    xa = mpl_copy[upper_stage_only == True] # X vectors
    y1a = 1./e1[upper_stage_only == True] # y1 = 1/e1
    y2a = 1./e2[upper_stage_only == True] # y2 = 1/e2
    # if mpl.size>1:
    
    # Find the critical payload mass separating the two curves
    # mpl_critical = max(mpl_copy[upper_stage_only])
    
    
    # Compute propellant masses
    mp1 = mE1/e1 - mE1 # Propellant mass 1st stage
    mp2 = mE2/e2 - mE2 # Propellant mass 2nd stage

    mptot = mp1+mp2

    # print("mpl size: " + str(mpl.size))
    # print("boolean result is: "+ str(mpl.size > 1) )
    
    return e1,e2,n,mp1,mp2,mptot

def long_term_mission(dv,Spcrft_Ve,Spcrft_Me,mE1,mp1_max,Isp1,mE2,mp2_max,Isp2,vbo, earth_emissions_per_kgram, MPE, concentration):
    
    import no_return_funcs as adv
    # standard function for finding mprop struggles to converge with higer dVs so the mba calculations require
    # the polyfit function to estimate reliably. 
    # this function needs a list however, so a dummy list is created here with the dv of interest as the first 
    # entry. 
    
    if np.size(Spcrft_Me)==1:
        dV = np.array([dv, dv+0.1, dv+.2])

        mpl = lunar_payload_mass(
            Spcrft_Me=Spcrft_Me, 
            Spcrft_Ve=Spcrft_Ve,
            dv=dV
            )

        mp1,mp2,mptot = adv.prop_mass_two_stage_polyfit(mE1,mp1_max,Isp1,mE2,mp2_max,Isp2,mpl,vbo)
        CO2, H2O, BC, NO2, GWP_BC, GWP_NO2, GWP_tot = long_term_mission_emissions(mptot)
        
        bep_mass, bep_time = break_even_point(GWP_tot, earth_emissions_per_kgram, MPE, Spcrft_Me, concentration)

        long_term_mission_data = {}
        long_term_mission_data['dv_to_destination'] = dv
        long_term_mission_data['payload_mass'] = mpl[0]
        long_term_mission_data['stage1_prop_mass'] = mp1[0]
        long_term_mission_data['stage2_prop_mass'] = mp2[0]
        long_term_mission_data['total_earth_launch_prop_mass'] = mptot[0]
        long_term_mission_data['GWP_CO2'] = CO2[0]
        long_term_mission_data['GWP_BC'] = GWP_BC[0]
        long_term_mission_data['GWP_NO2'] = GWP_NO2[0]
        long_term_mission_data['GWP_tot'] = GWP_tot[0]
        long_term_mission_data['bep_mass'] = bep_mass[0]
        long_term_mission_data['bep_time'] = bep_time[0]
        
    else:
        mpl = lunar_payload_mass(
            Spcrft_Me=Spcrft_Me, 
            Spcrft_Ve=Spcrft_Ve,
            dv=dv
            )
    
        mp1,mp2,mptot = adv.prop_mass_two_stage_polyfit(mE1,mp1_max,Isp1,mE2,mp2_max,Isp2,mpl,vbo)

        CO2, H2O, BC, NO2, GWP_BC, GWP_NO2, GWP_tot = long_term_mission_emissions(mptot)
        
        bep_mass, bep_time = break_even_point(GWP_tot, earth_emissions_per_kgram, MPE, Spcrft_Me, concentration)

        long_term_mission_data = {}
        long_term_mission_data['dv_to_moon'] = dv
        long_term_mission_data['payload_mass'] = mpl
        long_term_mission_data['stage1_prop_mass'] = mp1
        long_term_mission_data['stage2_prop_mass'] = mp2    
        long_term_mission_data['total_earth_launch_prop_mass'] = mptot
        long_term_mission_data['GWP_CO2'] = CO2
        long_term_mission_data['GWP_BC'] = GWP_BC
        long_term_mission_data['GWP_NO2'] = GWP_NO2
        long_term_mission_data['GWP_tot'] = GWP_tot
        long_term_mission_data['bep_mass'] = bep_mass[0]
        long_term_mission_data['bep_time'] = bep_time[0]

    return long_term_mission_data

def break_even_point(GWP_tot, earth_emissions_per_kgram, MPE, payload_mass, concentration):
    
    bep_mass = GWP_tot/earth_emissions_per_kgram

    bep_time = bep_mass/(MPE*payload_mass*concentration)

    return bep_mass, bep_time

def optimum_payload_mass(MPE, earth_emissions_per_kgram,dv,Spcrft_Ve,mE1,mp1_max,Isp1,mE2,mp2_max,Isp2,vbo,concentration, filename):
    import matplotlib.pyplot as plt

    payload_mass = np.linspace(0, 1e4, 10)

    long_term_mission_data=long_term_mission(dv,Spcrft_Ve,payload_mass,mE1,mp1_max,Isp1,mE2,mp2_max,Isp2,vbo,earth_emissions_per_kgram,MPE,concentration)
    
    bep_mass, bep_time = break_even_point(
        GWP_tot=long_term_mission_data['GWP_tot'], 
        earth_emissions_per_kgram=earth_emissions_per_kgram, 
        MPE=MPE, 
        payload_mass=payload_mass,
        concentration=concentration
    )

    print(bep_mass)
    print(bep_time)
    
    fig, axs = plt.subplots()

    # axs[0].scatter(payload_mass, bep_mass)
    # axs[0].set(xlabel = 'Payload Mass (kg)', ylabel = 'BEP Mass (kg')

    axs.scatter(payload_mass, bep_time)
    axs.set(xlabel = 'Payload Mass (kg)', ylabel = 'BEP Time')

    plt.tight_layout()
    plt.savefig('data/'+str(filename)+'LT_R')

def bep_vs_dV(Spcrft_Ve,payload_mass,mE1,mp1_max,Isp1,mE2,mp2_max,Isp2,vbo, earth_emissions_per_kgram,concentration, MPE):
    import pandas as pd

    dV_dict = {}
    for dV in np.linspace(3, 15, 13):

        long_term_mission_data=long_term_mission(dV,Spcrft_Ve,payload_mass,mE1,mp1_max,Isp1,mE2,mp2_max,Isp2,vbo,earth_emissions_per_kgram, MPE, concentration)
        
        bep_mass, bep_time=break_even_point(long_term_mission_data['GWP_tot'], earth_emissions_per_kgram, MPE, payload_mass, concentration)

        long_term_mission_data['bep_mass'] = bep_mass
        long_term_mission_data['bep_time'] = bep_time

        dV_dict[dV] = long_term_mission_data

    dataframe = pd.DataFrame.from_dict(dV_dict, orient='index')
    
    return dataframe

def return_mass(dv, Spcrft_Me, Spcrft_Ve, mex):

    MR = np.array(Spcrft_Me*(np.exp(-dv/Spcrft_Ve)-1)) + np.array(mex*np.exp(-dv/Spcrft_Ve))
        
    return np.array(MR)

def single_use_craft(dv,Spcrft_Ve,mass_transportation_craft,extracted_ISPP,earth_emissions_per_kgram,MPE,concentration,payload_mass, mE1,mp1_max,Isp1,mE2,mp2_max,Isp2,vbo):
    
    '''
    returns a dictionary with a key of the payload mass, each referencing a list:
    [extra_launches, gwp, new_return_mass, bep_mass, bep_time]
    '''

    # payload_mass = np.linspace(100, 1e4, 99)
    
    establishment_dataset = long_term_mission(dv,Spcrft_Ve,payload_mass,mE1,mp1_max,Isp1,mE2,mp2_max,Isp2,vbo,earth_emissions_per_kgram,MPE,concentration)
    establishment_gwp = establishment_dataset['GWP_tot']
    establishment_BC_gwp = establishment_dataset['GWP_CO2']

    return_mission_data = long_term_mission(dv,Spcrft_Ve,mass_transportation_craft,mE1,mp1_max,Isp1,mE2,mp2_max,Isp2,vbo,earth_emissions_per_kgram,MPE,concentration)
    return_gwp = return_mission_data['GWP_tot']
    return_BC_gwp = return_mission_data['GWP_BC']

    mass_returned = return_mass(dv, mass_transportation_craft, Spcrft_Ve, extracted_ISPP)
    print("mass returned is: "+str(mass_returned))
    
    launches_dict = {}

    for extra_launches in np.arange(20):
        
        dummy_dict = {}

        # print("extra launches: "+str(extra_launches))
        gwp = establishment_gwp + extra_launches*return_gwp
        BC_gwp = establishment_BC_gwp + extra_launches*return_BC_gwp

        new_return_mass = (extra_launches + 1)*mass_returned
        # print("return mass is: "+str(new_return_mass))

        
        bep_mass, bep_time = break_even_point(
                    GWP_tot=gwp, 
                    earth_emissions_per_kgram=earth_emissions_per_kgram, 
                    MPE=MPE, 
                    payload_mass=payload_mass,
                    concentration=concentration
                )

        ### finding the emission ratio after each return mission ------------------------------------------------
        
        emission_ratio = gwp/(new_return_mass*concentration*50)

        dummy_dict['gwp'] = gwp
        dummy_dict['GWP_BC'] = BC_gwp
        dummy_dict['return_mass'] = new_return_mass
        dummy_dict['bep_mass'] = bep_mass
        dummy_dict['bep_time'] = bep_time
        dummy_dict['emission_ratio'] = emission_ratio

        launches_dict[extra_launches] = dummy_dict

    return launches_dict

def single_use_spcraft_graphs(launches_dict, filename, fignum):
    import matplotlib.pyplot as plt
    import pandas as pd
    
    plt.figure(fignum)
    x = np.arange(20)
    df = pd.DataFrame.from_dict(launches_dict, orient='index')

    fig, axs = plt.subplots(nrows = 2)

    # axs[0].scatter(x, df['return_mass'])
    # axs[0].set(xlabel = 'Number of Launches', ylabel = 'Return Mass per mission')

    axs[0].scatter(x, df['bep_mass'])
    axs[0].set(xlabel = 'Number of Launches', ylabel = 'bep mass')

    axs[1].scatter(x, df['emission_ratio'])
    axs[1].set(xlabel = 'Number of Launches', ylabel = 'emission_ratio')

    plt.tight_layout()
    plt.savefig('graphs'+str(filename)+'LT_NR')

def SUS_BEP_graph(launches_dict, figname, earth_emissions_per_kg, concentration, ax, fignum):
    '''
    single use spacecraft, break-even point graph

    launches_dict from "single_use_craft()" 
    '''
    import pandas as pd
    
    # i want to see the return mass vs gwp on the same graph as terrestrial emissions
    df = pd.DataFrame.from_dict(launches_dict, orient='index')

    return_mass = np.concatenate(([0], df['return_mass']))
    return_mass = return_mass[:-1] # removes the last element 
    plat_returned = return_mass*concentration

    ax[fignum].plot(plat_returned, df['gwp'], 'b', label='ETM Emissions')
    earth_emissions = plat_returned*earth_emissions_per_kg 
    ax[fignum].plot(plat_returned, earth_emissions, 'g', label='Terr. Emissions') 
    ax[fignum].title.set_text(figname)

def optimum_payload_mass_graph(launches_dict, figname, earth_emissions_per_kg, MPE, concentration, ax, fignum):

    payload_mass = np.linspace(0, 1e4, 100)
    
    bep_mass, bep_time = break_even_point(
        GWP_tot=launches_dict['GWP_tot']-launches_dict['GWP_BC'], 
        earth_emissions_per_kgram=earth_emissions_per_kg, 
        MPE=MPE, 
        payload_mass=payload_mass,
        concentration=concentration
    )

    ax[fignum].scatter(payload_mass, bep_time)
    ax[fignum].title.set_text(figname)
    

    


