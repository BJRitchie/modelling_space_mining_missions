import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math 

def dV_SH(df):
        '''
        Compute the Shoemaker-Helin delta-V (and its comonents) from a set of 
        orbital elements.
        
        Delta-V is computed following the approach described by:
        Shoemaker and Helin (1978), Earth-approaching asteroids as targets for 
        exploration, NASA CP-2053, pp. 245-256.
        
        Code adapted from Benner: http://echo.jpl.nasa.gov/~lance/delta_v/delta_v.rendezvous.html
        See: http://echo.jpl.nasa.gov/~lance/delta_v/deltav.13.pl
        
        Parameters
        ----------
        df : pandas datafram
            Dataframe containing orbital elements (a,e,i,apo,peri)

        Returns
        -------
        df : pandas datafram
            Original dataframe with Shoemaker-Helin delta-V (and its components) 
            appended.

        '''
        
        # Constants -----------------------------------------------------------
        #vearth = 30.0	    # Orbital speed of Earth in km/s (used by Shoemaker)
        vearth = 29.784		# More accurate orbital speed of Earth in km/s
        
        # TODO: make U0 a function of input orbtial altitude
        #U0 = 7.907/vearth	# Normalized Orbital velocity at Earth's surface 
        U0 = 7.786/vearth	# Normalized Orbital velocity at an altitude of 200 km
        
        S = np.sqrt(2)*U0	# Earth's normalized escape velocity

        # delta_v_mars = 6.3	# LEO -> Mars rendezvous.  Units: km/s
        # delta_v_moon = 6.0	# LEO -> Lunar surface.   Units: km/s
        # aj = 5.2026         # Semimajor axis of Jupiter
    
        
        # Compute Departure and Arrival manouvres -----------------------------
        
        # Use Split-Apply-Combine approach to compute the delta-Vs of the 
        # individual groups
        # In each group, extract the individual elements to simplify the 
        # representations of the equations. 
        
        
        # Apollos
        dfapollos = df[(df.q <= 1) & (df.a >= 1)]
        a = dfapollos.a            # Semi-major axis (AU)
        e = dfapollos.e            # Eccentricity
        i = dfapollos.i*np.pi/180  # Inclination (rad)
        Q = dfapollos.Q          # Aphelion distance (AU)
        q = dfapollos.q         # Perihelion distance (AU)
        dfapollos['ut2'] = 3 - 2/(Q+1) - 2*np.cos(i/2)*np.sqrt(2*Q/(Q+1))
        dfapollos['uc2'] = 3/Q - 2/(Q+1) - (2/Q)*np.sqrt(2/(Q+1))
        dfapollos['ur2'] = 3/Q - 1/a - (2/Q)*np.cos(i/2)*np.sqrt((a/Q)*(1-e**2))
        del a, e, i, Q, q
        
        # Amors
        dfamors = df[(df.q > 1) & (df.a >= 1)]
        a = dfamors.a            # Semi-major axis (AU)
        e = dfamors.e            # Eccentricity
        i = dfamors.i*np.pi/180  # Inclination (rad)
        Q = dfamors.Q          # Aphelion distance (AU)
        q = dfamors.q         # Perihelion distance (AU)
        dfamors['ut2'] = 3 - 2/(Q+1) - 2*np.cos(i/2)*np.sqrt(2*Q/(Q+1))
        dfamors['uc2'] = 3/Q - 2/(Q+1) - (2/Q)*np.cos(i/2)*np.sqrt(2/(Q+1))
        dfamors['ur2'] = 3/Q - 1/a - (2/Q)*np.sqrt(a*(1-e**2)/Q)
        del a, e, i, Q, q
        
        # Atens
        dfatens = df[df.a < 1]
        a = dfatens.a            # Semi-major axis (AU)
        e = dfatens.e            # Eccentricity
        i = dfatens.i*np.pi/180  # Inclination (rad)
        Q = dfatens.Q          # Aphelion distance (AU)
        q = dfatens.q         # Perihelion distance (AU)
        dfatens['ut2'] = 2 - 2*np.cos(i/2)*np.sqrt(2*Q - Q**2)
        dfatens['uc2'] = 3/Q - 1 - (2/Q)*np.sqrt(2 - Q)
        dfatens['ur2'] = 3/Q - 1/a - (2/Q)*np.cos(i/2)*np.sqrt(a*(1-e**2)/Q) 
        del a, e, i, Q, q
        # TODO: add alternative method for periapsis transfer
        
        
        # Compute Detal-Vs ----------------------------------------------------
        
        # Combine groups into single dataframe
        df = pd.concat((dfatens, dfapollos, dfamors))
        
        # Fix any negative velocities
        df['ur2'][df.ur2 < 0] = abs(df.ur2)
        df['uc2'][df.uc2 < 0] = abs(df.uc2)
        df['ut2'][df.ut2 < 0] = abs(df.ut2)

        # Compute manouvres UL & UR (normalized velocity unitys)
        i = df.i*np.pi/180  # Inclination (rad)
        UL = np.sqrt(df.ut2 + S**2) - U0                            # Earth departure
        ur = np.sqrt(df.ur2)
        uc = np.sqrt(df.uc2)
        UR = np.sqrt(df.uc2 + df.ur2 - 2*ur*uc*np.cos((df.i*np.pi/180)/2) ) # Ast arrival
    
        
        # Figure of merit (delta-V in normalized velocity units)
        F = UL + UR # Shoemaker's figure of merit
        
        # Delta-Vs
        dV_SH = (30*F + 0.5)    # Delta-V for a rendezvous (km/s)
        dV_SH1 = (30*UL + 0.5)  # Departure manouvre (Delta-V for a flyby) (km/s)
        dV_SH2 = (30*UR + 0.5)  # Arrival manouvre (km/s)
        
        # Append to output Dataframe ------------------------------------------
        df['dV_SH'] = dV_SH
        df['dV_SH1'] = dV_SH1
        df['dV_SH2'] = dV_SH2
        
        df = df.sort_index()
        
        return df

def prop_mass_single_stage(mE, mpl, Isp,vbo):
    '''
    Compute the propellant mass for a single-stage launch vehicle to reach a
    specified burnout velocity (of delta-V).

    Parameters
    ----------
    mE : float
        Empty mass of launch vehicle (kg)
    mpl : float or nx1 array
        Payload mass (kg)
    Isp : float
        Specific impulse (s)
    vbo : float or nx1 array
        Burnout velocity (km/s)

    Returns
    -------
    mp : TYPE
        Propellant mass

    '''
    
    # Compute exhaust velocity
    ve = Isp*9.81*1e-3 # Exhaust velocity (km/s)
    
    mp_ss = (np.array(mE) + np.array(mpl))*(np.exp(vbo/ve) - 1) # Single stage
    return mp_ss

def prop_mass_two_stage(mE1,mp1_max,Isp1,mE2,mp2_max,Isp2,mpl,vbo):
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
    import numpy.polynomial.polynomial as poly
    
    
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
    mp2_vec = prop_mass_single_stage(mE2,mpl,Isp2,vbo) # Propellant masses
    upper_stage_only = mp2_vec < mp2_max
    # Upper_stage_only flag:
    # True : payload can be launched with just the upper stage.
    #        Set 1st stage as empty e1 = 0; e2 = 1 as initial guess
    # False: need both stages to reach orbit.
    #        set 
    
    # Loop through
    
    print('prop_mass_two_stage: solving equations')

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
    if True:
        # Fit polynomial equations to the data
        coefs1a = poly.polyfit(xa, y1a, 1) # Coeffs of curve y1 = 1/e1 = a0 + a1*x
        coefs2a = poly.polyfit(xa, y2a, 1) # Coeffs of curve y2 = 1/e2 = a0 + a1*x
        # Fill any missing values with the those predicted from the polynomials
        ind = (np.isnan(e1)) & (upper_stage_only == True)
        e1[ind] = 1./poly.polyval( mpl_copy[ind],coefs1a  ) # Take 1/value since the curve is 1/e1
        ind = (np.isnan(e2)) & (upper_stage_only == True)
        e2[ind] = 1./poly.polyval( mpl_copy[ind],coefs2a  )
        
        # b. Both stages
        # Extract data
        ind = (upper_stage_only == False) & ~np.isnan(e1)
        xb = mpl_copy[ind] # X vectors
        y1b = 1./e1[ind] # y1 = 1/e1
        y2b = 1./e2[ind] # y2 = 1/e2
        
        # Fit polynomial equations to the data
        coefs1b = poly.polyfit(xb, y1b, 1) # Coeffs of curve y1 = 1/e1 = a0 + a1*x + a2*x^2
        coefs2b = poly.polyfit(xb, y2b, 1) # Coeffs of curve y2 = 1/e2 = a0 + a1*x + a2*x^2
        # Fill any missing values with the those predicted from the polynomials
        ind = (np.isnan(e1)) & (upper_stage_only == False)
        e1[ind] = 1./poly.polyval( mpl_copy[ind],coefs1b  )
        ind = (np.isnan(e2)) & (upper_stage_only == False)
        e2[ind] = 1./poly.polyval( mpl_copy[ind],coefs2b  )
    
    # Find the critical payload mass separating the two curves
    mpl_critical = max(mpl_copy[upper_stage_only])
    
    
    # Compute propellant masses
    mp1 = mE1/e1 - mE1 # Propellant mass 1st stage
    mp2 = mE2/e2 - mE2 # Propellant mass 2nd stage

    mptot = mp1+mp2

    print("mpl size: " + str(mpl.size))
    print("boolean result is: "+ str(mpl.size > 1) )
    
    return e1,e2,n,mp1,mp2,mptot,coefs1a,coefs2a,coefs1b,coefs2b, mpl_critical
    
def prop_mass_two_stage_polyfit(mE1,mp1_max,Isp1,mE2,mp2_max,Isp2,mpl,vbo):
    ''' 
    Compute propellant mass for 2-stage launch vehicle.
    
    This method first uses the prop_mass_two_stage() over a smaller list of
    payload masses to produce polynomial trend lines. It then uses these trend
    lines to find the propellant masses for the requested payload masses.

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
    mp1 : TYPE
        Propellant mass of 1st stage (kg)
    mp2 : TYPE
        Propellant mass of 2nd stage (kg)
    mp : TYPE
        Total propellant mass of both stages (kg)
    
    '''
    
    import numpy.polynomial.polynomial as poly
    
    # Finding the polynomial fit ----------------------------------------------
    
    # Generate a new list of mpl values and compute the propellant masses
    # using the prop_mass_two_stage() function.
    
    mpl_new = np.linspace(1000,60000,1000) # Payload mass (kg)
    # print(mpl_new)
    
    # Call the function to return the polynomials 
    e1,e2,n,mp1,mp2,mptot,coefs1a,coefs2a,coefs1b,coefs2b, mpl_critical = prop_mass_two_stage(mE1,mp1_max,Isp1,mE2,mp2_max,Isp2,mpl_new,vbo)
    
    print('prop_mass_two_stage_polyfit: coefficients found')
    
    # Delete these values
    del e1,e2,n,mp1,mp2,mptot
    
    # Use the coeffs to predict the curve -------------------------------------
    
    # Create empty list of e1,e2

    if np.size(mpl)>1:
        e1 = np.zeros(len(mpl))*np.nan
        e2 = np.zeros(len(mpl))*np.nan
        
        # Upper stage only
        ind = mpl <= mpl_critical # Indices of values
        e1[ind] = 1./poly.polyval( mpl[ind],coefs1a  )
        e2[ind] = 1./poly.polyval( mpl[ind],coefs2a  )
        
        # Both stages
        ind = mpl > mpl_critical # Indices of values
        e1[ind] = 1./poly.polyval( mpl[ind],coefs1b  )
        e2[ind] = 1./poly.polyval( mpl[ind],coefs2b  )
        
        # Compute propellant masses
        mp1 = mE1/e1 - mE1 # Propellant mass 1st stage
        mp2 = mE2/e2 - mE2 # Propellant mass 2nd stage
        mptot = mp1 + mp2 # Total propellant mass
    
    elif np.size(mpl)==1:
        # Upper stage only
        e1 = 1./poly.polyval( mpl,coefs1a  )
        e2 = 1./poly.polyval( mpl,coefs2a  )
        
        # Both stages
        e1 = 1./poly.polyval( mpl,coefs1b  )
        e2 = 1./poly.polyval( mpl,coefs2b  )
        
        # Compute propellant masses
        mp1 = mE1/e1 - mE1 # Propellant mass 1st stage
        mp2 = mE2/e2 - mE2 # Propellant mass 2nd stage
        mptot = mp1 + mp2 # Total propellant mass
    
    print('prop_mass_two_stage_polyfit: propellant mass found')
    
    return mp1,mp2,mptot

def test_two_stage_propellant_mass():
    '''
    Main Script to test the propellant mass functions
    '''
    import matplotlib.pyplot as plt

    # Inputs ------------------------------------------------------------------
    
    # Delta-V to 200 km orbit
    vbo = 7.58 #
    
    # Vector of payloads
    mpl = np.linspace(1000,60000,1000000) # Payload mass (kg)
    
    # Single stage rocket parameters
    Isp = 282.   # Specific impulse
    g0 = 9.81*1e-3
    ve = g0*Isp # Exhaust velocity
    mE = (22.2 + 4 + 1.7)*1000 # Total dry mass (kg) (from wiki)
    mp_total = 1420000 - mE    # Total propellant mass capacity
    
    # Multi-stage parameters
    mE1 =  (22.2 +1.7)*1000 # Empty mass of 1st stage (kg) (Including fairing)
    mp1_max = 410.9*1000    # Max propellant capacity of 1st stage (kg)
    Isp1 = 282.             # Specific impulse of 1st stage
    mE2 =  4*1000           # Empty mass of 2nd stage (kg)
    mp2_max = 107.5*1000    # Max propellant capacity of 2nd stage (kg)
    Isp2 = 311 # Specific impulse of 2nd stage
    
    
    # Computations ------------------------------------------------------------
    
    # Single-stage rocket
    mp_ss = prop_mass_single_stage(mE,mpl,Isp,vbo)
    # mp_ss = (mE + mpl)*(np.exp(dV/ve) - 1) # Total propellant mass (kg)
    
    # Two-stage rocket 
    # # Using full numerical method
    # e1,e2,n,mp1,mp2,coefs1a,coef2a,coefs1b,coefs2b,mpl_critical = prop_mass_two_stage(mE1,mp1_max,Isp1,mE2,mp2_max,Isp2,mpl,dV)
    # Using polynomial fit
    mp1,mp2,mp = prop_mass_two_stage_polyfit(mE1,mp1_max,Isp1,mE2,mp2_max,Isp2,mpl,vbo)
    
    
    
    mp = mp1+mp2 # Total propellant for two-stage
    
    # Plots -------------------------------------------------------------------
    
    # fig = plt.figure()
    # plt.xlabel('Payload Mass (kg)', fontsize=10)
    # plt.ylabel('Propellant mass (kg)', fontsize=10)
    # plt.plot(mpl,mp_ss,'k', label='Single-stage')
    # plt.plot(mpl,mp1+mp2,'r', label='Two-stage total')
    # plt.plot(mpl,mp1, label='1st stage')
    # plt.plot(mpl,mp2, label='2nd stage')
    # plt.legend(loc='upper left')
    
    return mpl, mp_ss, mp1, mp2, mp

def payload_mass(df, Spcrft_Me, Spcrft_Ve):
    '''
    Payload mass upon launch ------------------------------------------------------------------------
    This mass is the only variable that changes for the launch and is dependent upon the dV required 
    to reach the desired asteroid
    
    '''
    Spcrft_mp = Spcrft_Me*(np.exp(df['dV_SH']/Spcrft_Ve)-1)
    mpl =  np.array(Spcrft_mp) + np.array(Spcrft_Me)
    df['mpl'] = mpl
    
    print('payload_mass: payload mass found')
    
    return mpl

def return_mass(df, Spcrft_Me, Spcrft_Ve, mex):
    '''
    This is crucial to determining the emissions per kg returned, but has has no effect on the 
    launch mass as all fuel required return is mined on site
    '''
    
    dv1 = df['dV_SH']
    
    MR = np.array(Spcrft_Me*(np.exp(-dv1/Spcrft_Ve)-1)) + np.array(mex*np.exp(-dv1/Spcrft_Ve))
    df['MR'] = MR
    
    print('return_mass: return mass found')
    
    return np.array(MR)

def emissions(df, EICO2, EIH2O, EIBC, EINO2, 
              mE1,mp1_max,Isp1,mE2,mp2_max,Isp2,vbo, 
              Spcrft_Me, Spcrft_Ve, mex):
    
    #### Variables - from "Radiative_Forcing_Caused_by_Rocket_Engine_Emission.pdf" MN Ross
    
    ### GWP values - CO2e per kg component
    GWP_CO2_ = 1
    GWP_BC_ = 50000
    GWP_NO2_ = 281.5
    
    ### kgs of each component emitted
    # multiplied by 2/3 because this is the average amount of fuel burnt above the troposphere in a rocket launch 
    # (Ross's paper^)
    
    mpl = payload_mass(df, Spcrft_Me, Spcrft_Ve)
    print("In emissions func: mpl is " + str(mpl.size))
    print('emissions: payload mass imported')
    
    mp1,mp2,mprop = prop_mass_two_stage_polyfit(mE1,mp1_max,Isp1,mE2,mp2_max,Isp2,mpl,vbo)
    print('emissions: propellant mass calculated')
    df['mPropStage1'] = mp1
    df['mPropStage2'] = mp2
    df['mPropTotal'] = mprop
    
    MR = return_mass(df, Spcrft_Me, Spcrft_Ve, mex)
    print('emissions: return mass calculated')

    CO2 = EICO2 * mprop
    df['CO2']= CO2

    H2O = EIH2O * (2/3) * mprop
    df['H2O']= H2O

    BC = EIBC * (2/3) * mprop
    df['BC']= BC
    
    NO2 = EINO2 * mprop
    df['NO2'] = NO2
    
    ### NO2 emissions - predicted that 20% of the mass is released in the form of NO2
    NO2_2 = (np.array(df['MR']) + np.array(Spcrft_Me)) * 0.2 # return to earth

    ### Global Warming Potential for each component
    GWP_BC = GWP_BC_ * BC
    df['CO2e_BC'] = GWP_BC

    GWP_NO2 = GWP_NO2_ * (np.array(NO2) + np.array(NO2_2))
    df['GWP_NO2'] = GWP_NO2

    GWP_tot = np.array(CO2) + np.array(GWP_BC) + np.array(GWP_NO2)
    df['CO2e_tot'] = GWP_tot
    

    # calculating co2e emitted per kg returned
    concentration = ((184.5)*10**(-6))*50

    ratio = GWP_tot/(MR*concentration)
    df['Ratio'] = ratio
    print('emissions: ratios found')
    
    return ratio, CO2, H2O, BC, NO2, GWP_BC, GWP_NO2, GWP_tot

def mba_orbital_elements():
#     amor = pd.read_csv('Downloads/Science_Ext/amor.csv')
#     Qa = amor['a']*(1 + amor['e'])
#     amor['Q'] = Qa

#     apollo = pd.read_csv('Downloads/Science_Ext/Apollo.csv')
#     Qb = apollo['a']*(1 + apollo['e'])
#     apollo['Q'] = Qb

#     aten = pd.read_csv('Downloads/Science_Ext/Aten.csv')
#     Qc = aten['a']*(1 + aten['e'])
#     aten['Q'] = Qc

    mba = pd.read_csv('Downloads/Science_Ext/MBA.csv')
    Qd = mba['a']*(1 + mba['e'])
    mba['Q'] = Qd

#     df = pd.concat((amor, apollo, aten)).reset_index(drop=True)
    df2 = dV_SH(mba)
    
    return df2

def extracted_mass():
    # # Defining fuel capacity of transportation spacecraft -----------------------------------------------------
    # Volume Available
    height = 6.7 # before tapering
    rad = 2.3 
    vol = np.pi * rad**2 * height # ~ 111
    avail_vol = 0.8 * vol # assuming 80% of the volume can be used for fuel = ~90m3

    # Densities etc
    Burn_Ratio = 5.35 # average of engines on wiki
    Density_LH2 = 71 # kg/m3
    Density_LOx = 1.141e+3 # kg/m3
    Vol_Ratio = 0.333 #LOx : LH2
    Pcnt_LOx = 0.25 # vol
    Pcnt_LH2 = 0.75 # vol

    LOx = Pcnt_LOx * avail_vol * Density_LOx # ~ 2.54e+4
    LH2 = Pcnt_LH2 * avail_vol * Density_LH2 # ~ 4.74e+3

    mex =  LOx + LH2 # ~ 3.02e+4
    
    print('mex: extracted mass found')
    
    return mex

# def test_ratio():
    '''
    Main Script to test the propellant mass functions
    '''
    
    # Inputs ------------------------------------------------------------------
    
    df = orbital_elements()
    
    Spcrft_Me = 1000
    Spcrft_Ve = 0.0098*444.2 # average of engines on wiki

    # Fuel Required to Leave Atmosphere ---------------------------------------------------------------------------
    # Delta-V to 200 km orbit
    vbo = 7.58 #

    # Vector of payloads
    #     mpl = np.linspace(1000,60000,1000) # Payload mass (kg)

    # Single stage rocket parameters
    Isp = 282.   # Specific impulse
    g0 = 9.81*1e-3
    ve = g0*Isp # Exhaust velocity
    mE = (22.2 + 4 + 1.7)*1000 # Total dry mass (kg) (from wiki)
    mp_total = 1420000 - mE    # Total propellant mass capacity
    mex = extracted_mass()

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

    
    # Computations ------------------------------------------------------------
    
    df = orbital_elements()
    print('test_ratio: orbital elements downloaded')
    
    mpl = payload_mass(df, 
                       Spcrft_Me = 1000, 
                       Spcrft_Ve = 0.0098*444.2)
    print('test_ratio: payload mass imported')
    
    Ratio = emissions(df, 
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
                      Spcrft_Me = 1000, 
                      Spcrft_Ve = 0.0098*444.2, 
                      mex = extracted_mass() )
    
    print('test_ratio: Ratio calculated')
    
    
    MR = df['MR']


    
    
    # Plots -------------------------------------------------------------------
    
    fig = plt.figure(figsize = (10,5))
    plt.xlabel('Return Mass (kg)', fontsize=10)
    plt.ylabel('Ratio', fontsize=10)
    plt.scatter(MR,ratio, c = 'r' #label='Single-stage'
               )
#     plt.plot(mpl,mp1+mp2,'r', label='Two-stage total')
#     plt.plot(mpl,mp1, label='1st stage')
#     plt.plot(mpl,mp2, label='2nd stage')
#     plt.legend(loc='upper left')
    
    return

def mba_generate_df():
    # Input into dataframe 
    
    df = mba_orbital_elements()
    
    def payload():
        
        mpl = payload_mass(df, 
               Spcrft_Me = 1000, 
               Spcrft_Ve = 0.0098*444.2)
        
        return mpl
    
    mpl = payload()
    df['payload_mass'] = mpl
    
    
    mp1,mp2,mp = prop_mass_two_stage_polyfit(mE1 =  (22.2 +1.7)*1000,
                                             mp1_max = 410.9*1000,
                                             Isp1 = 282 ,
                                             mE2 =  4*1000,
                                             mp2_max = 107.5*1000 ,
                                             Isp2 = 311,
                                             mpl = payload(),
                                             vbo = 7.58)
    df['Mprop_Lower_Stage'] = mp1
    df['Mprop_Upper_Stage'] = mp2
    df['Mprop_Total'] = mp

    
    
    
    ratio, CO2, H2O, BC, NO2, GWP_BC, GWP_NO2, GWP_tot = emissions(df, 
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
                                                                  Spcrft_Me = 1000, 
                                                                  Spcrft_Ve = 0.0098*444.2, 
                                                                  mex = extracted_mass() )
    
    
    df['CO2(kg)'] = CO2
    df['H2O(kg)'] = H2O
    df['Black_Carbon(kg)'] = BC
    df['NO2(kg)'] = NO2
    df['CO2e_BC'] = GWP_BC
    df['CO2e_NO2'] = GWP_NO2
    df['CO2e_tot'] = GWP_tot
    
    return df

def proportion_beneficial(df,b):
    
            
    below = len(df[(df['Ratio'] < 4.18e+4/10**b)])

    total = len(df['Ratio'])
    
    percent = (below/total) * 100
        
        
    
    return below, total, percent

def proportion_beneficial_func(data, b, kind):
    
    if kind == 'BC':
        
        total = len(data['Ratio'])
        
        array = data['Ratio']
        
    
    
    ### adjusting to create graphs that don't include the effects of BC for comparison:
    
    if kind == 'no_BC':
        
        co2e_noBC = np.array(data['CO2e_NO2']) + np.array(data['CO2(kg)'])
        ratio_noBC = data['MR']/co2e_noBC
        
        total = len(ratio_noBC)
        
        array = ratio_noBC
        
        
        
        
    arr1 = np.linspace(-10, 2, b)

    below = np.empty_like(arr1)

    percent = np.empty_like(arr1)

    x = 0

    while x < b:

        below[x] = len(array[array < (4.18e+4/10**(-arr1[x]))])

        percent[x] = below[x]/total * 100

        x = x+1
    
    
    return arr1, below, percent


# def generate_df():
#     # Input into dataframe 
    
#     df = orbital_elements()
    
#     def payload():
        
#         mpl = payload_mass(df, 
#                Spcrft_Me = 1000, 
#                Spcrft_Ve = 0.0098*444.2)
        
#         return mpl
    
#     mpl = payload()
#     df['payload_mass'] = mpl
    
    
    
# #     mp_ss = prop_mass_single_stage(df, 
# #                                    mE = (22.2 + 4 + 1.7)*1000,
# #                                    Isp = 282,
# #                                    vbo = 7.58)
# #     df['Single_Stage_Mp'] = mp_ss
    
    
#     mp1,mp2,mp = prop_mass_two_stage_polyfit(mE1 =  (22.2 +1.7)*1000,
#                                              mp1_max = 410.9*1000,
#                                              Isp1 = 282 ,
#                                              mE2 =  4*1000,
#                                              mp2_max = 107.5*1000 ,
#                                              Isp2 = 311,
#                                              mpl = payload(),
#                                              vbo = 7.58)
#     df['Mprop_Lower_Stage'] = mp1
#     df['Mprop_Upper_Stage'] = mp2
#     df['Mprop_Total'] = mp

    
    
    
#     ratio, CO2, H2O, BC, NO2, GWP_BC, GWP_NO2, GWP_tot = emissions(df, 
#                                                                   EICO2 = 0.65, 
#                                                                   EIH2O = 0.3, 
#                                                                   EIBC = 0.02, 
#                                                                   EINO2 = 0.02, 
#                                                                   mE1 =  (22.2 +1.7)*1000,
#                                                                   mp1_max = 410.9*1000,
#                                                                   Isp1 = 282 ,
#                                                                   mE2 =  4*1000,
#                                                                   mp2_max = 107.5*1000 ,
#                                                                   Isp2 = 311,
#                                                                   vbo = 7.58, 
#                                                                   Spcrft_Me = 1000, 
#                                                                   Spcrft_Ve = 0.0098*444.2, 
#                                                                   mex = extracted_mass() )
    
    
#     df['CO2(kg)'] = CO2
#     df['H2O(kg)'] = H2O
#     df['Black_Carbon(kg)'] = BC
#     df['NO2(kg)'] = NO2
#     df['CO2e_BC'] = GWP_BC
#     df['CO2e_NO2'] = GWP_NO2
#     df['CO2e_tot'] = GWP_tot
    
#     return df