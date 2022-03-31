import numpy as np


muabo = np.genfromtxt("./muabo.txt", delimiter=",")
muabd = np.genfromtxt("./muabd.txt", delimiter=",")

red_wavelength = 600 # Replace with wavelength in nanometres
green_wavelength = 515 # Replace with wavelength in nanometres
blue_wavelength = 460 # Replace with wavelength in nanometres

wavelength = np.array([red_wavelength, green_wavelength, blue_wavelength])

def mua_blood_oxy(x): return np.interp(x, muabo[:, 0], muabo[:, 1])
def mua_blood_deoxy(x): return np.interp(x, muabd[:, 0], muabd[:, 1])

bvf = 0.01 # Blood volume fraction, average blood amount in tissue
oxy = 0.8 # Blood oxygenation

# Absorption coefficient ($\mu_a$ in lab text)
# Units: 1/m

mua_other = 25 # Background absorption due to collagen, et cetera
mua_blood = (mua_blood_oxy(wavelength)*oxy # Absorption due to
            + mua_blood_deoxy(wavelength)*(1-oxy)) # pure blood
mua = mua_blood*bvf + mua_other

# reduced scattering coefficient ($\mu_s^\prime$ in lab text)
# the numerical constants are thanks to N. Bashkatov, E. A. Genina and
# V. V. Tuchin. Optical properties of skin, subcutaneous and muscle
# tissues: A review. In: J. Innov. Opt. Health Sci., 4(1):9-38, 2011.
# Units: 1/m
musr = 100 * (17.6*(wavelength/500)**-4 + 18.78*(wavelength/500)**-0.22)

# mua and musr are now available (as shape (3,) arrays
# Red, green and blue correspond to indexes 0, 1 and 2, respectively

# TODO calculate penetration depth
pen_depth = (1/(3*(musr+mua)*mua))**0.5
print(f"penetrationdepth : {pen_depth} [m]")
d=300*10**(-6)

def cal_light_Trans(z,mua,musr):
    phi_zero = 1/(2*pen_depth*mua)
    C = np.sqrt(3*mua*(musr+mua))
    phi_pos = phi_zero * np.exp(-C*z)
    T = phi_pos/phi_zero 
    T_prosent = T*100
    #print(C)
    T_2 = np.exp(-C*z)
    print(T_2)
    
    return T_2*100


print(f"{cal_light_Trans(d,mua,musr)} ") # transmittans

def cal_light_R(z,pen_depth,mua,musr):
    T = cal_light_Trans(z,mua , musr)/100
    R = 1-T
    return R

#print(cal_light_R(0.015,pen_depth,mua,musr))

 


def cal_T_bloodfrac(d,mua,musr):
    C = np.sqrt(3*mua*(musr+mua))
    T = np.exp(-C*d)   
    return T


#print(cal_T_bloodfrac(d,mua,musr))


def cal_mu(bvf):
    red_wavelength = 600 # Replace with wavelength in nanometres
    green_wavelength = 515 # Replace with wavelength in nanometres
    blue_wavelength = 460 # Replace with wavelength in nanometres

    wavelength = np.array([red_wavelength, green_wavelength, blue_wavelength])

    def mua_blood_oxy(x): return np.interp(x, muabo[:, 0], muabo[:, 1])
    def mua_blood_deoxy(x): return np.interp(x, muabd[:, 0], muabd[:, 1])

 
    oxy = 0.8 # Blood oxygenation

    # Absorption coefficient ($\mu_a$ in lab text)
    # Units: 1/m

    mua_other = 25 # Background absorption due to collagen, et cetera
    mua_blood = (mua_blood_oxy(wavelength)*oxy # Absorption due to
                + mua_blood_deoxy(wavelength)*(1-oxy)) # pure blood
    mua = mua_blood*bvf + mua_other

    # reduced scattering coefficient ($\mu_s^\prime$ in lab text)
    # the numerical constants are thanks to N. Bashkatov, E. A. Genina and
    # V. V. Tuchin. Optical properties of skin, subcutaneous and muscle
    # tissues: A review. In: J. Innov. Opt. Health Sci., 4(1):9-38, 2011.
    # Units: 1/m
    musr = 100 * (17.6*(wavelength/500)**-4 + 18.78*(wavelength/500)**-0.22)

    return mua,musr



def cal_K(d) : 
    mua_l , musr_l = cal_mu(0.01)
    mua_h , musr_h = cal_mu(1)
    T_l = cal_T_bloodfrac(d,mua_l,musr_l)
    T_h = cal_T_bloodfrac(d,mua_h,musr_h)
    
    K = np.abs(T_h-T_l)/(T_l)
    
    return K
    
print(cal_K(d))