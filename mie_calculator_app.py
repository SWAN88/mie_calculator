import pandas as pd
import numpy as np
import streamlit as st
import time
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d
import plotly.express as px
import plotly.graph_objects as go
from matplotlib import cm
from tqdm import tqdm
from cmath import pi
from scipy.special import *
from bokeh.plotting import figure
from scipy.special import spherical_jn, spherical_yn

"""

# :full_moon: :chart_with_upwards_trend: Mie Calculator

Here you can calculate the optical cross-sections of spherical nanoparticles using Mie theory :heart:.

***Instructions***

Select a material from the dropdown menu and enter the radius in nanometers. 

Enter the refractive index of the surrounding media.

You can see the calculation results of the extinction, absorption and scattering cross sections immediately.

A csv file of the data can be downloaded at the end.

"""
# define the a_n and b_n coefficients in Yigao's way
def spherical_hn(n, z):
    return spherical_jn(n, z) + 1j*spherical_yn(n, z)


def a_n(m, n, x):
    miu = 1
    a1 = m**2*spherical_jn(n, m*x)*(x*spherical_jn(n-1, x)-n*spherical_jn(n, x))
    a2 = miu*spherical_jn(n, x)*(m*x*spherical_jn(n-1, m*x) -n*spherical_jn(n, m*x))
    a3 = m**2*spherical_jn(n, m*x)*(x*spherical_hn(n-1, x)-n*spherical_hn(n, x))
    a4 = miu*spherical_hn(n, x)*(m*x*spherical_jn(n-1, m*x) -n*spherical_jn(n, m*x))
    return (a1-a2)/(a3-a4)


def b_n(m, n, x):
    miu = 1
    b1 = miu*spherical_jn(n, m*x)*(x*spherical_jn(n-1, x)-n*spherical_jn(n, x))
    b2 = spherical_jn(n, x)*(m*x*spherical_jn(n-1, m*x) -n*spherical_jn(n, m*x))
    b3 = miu*spherical_jn(n, m*x)*(x*spherical_hn(n-1,x)-n*spherical_hn(n, x))
    b4 = spherical_hn(n, x)*(m*x*spherical_jn(n-1, m*x) -n*spherical_jn(n, m*x))
    return (b1-b2)/(b3-b4)


def mie(m, x, n_max, radius):
    c_0 = 2*pi / (x/radius)**2
    c_sca = 0 # np.zeros([n_max])
    c_ext = 0 # np.zeros([n_max])
    for i in range(1, n_max+1):
        a_i = a_n(m, i, x)
        b_i = b_n(m, i, x)
        c_sca += c_0*(2*i+1)*(np.absolute(a_i)**2+np.absolute(b_i)**2)
        c_ext += c_0*(2*i+1)*np.real(a_i+b_i)
    return c_sca, c_ext


def func_pin(n, cth):
    if (n<0) or (n != int(n)): 
        print('n must be positive integer') 
        return -1
    elif n==0: return 0
    elif n==1: return 1
    else: 
        return (2*n-1)/(n-1)*cth*func_pin(n-1,cth) - n/(n-1)*func_pin(n-2,cth)

    
def func_taon(n, cth):
    if (n<0) or (n != int(n)): 
        print('n must be positive integer') 
        return -1
    elif n==0: return 0
    elif n==1: return cth
    else: 
        return n*cth*func_pin(n,cth) - (n+1)*func_pin(n-1,cth)        

    
def M_3oln(r, theta, phi, m, k, n):
    cth = np.cos(theta)
    z = r*m*k  # m==1 , z is unitless.
    M1 = 0 #np.zeros(len(x))
    M2 = np.cos(phi)*func_pin(n, cth)*spherical_hn(n, z)
    M3 = -np.sin(phi)*func_taon(n, cth)*spherical_hn(n, z)
    return M1, M2, M3


def N_3eln(r, theta, phi, m, k, n):
    cth = np.cos(theta)
    z = r*m*k
    N1 = n*(n+1)*np.cos(phi)*np.sin(theta)*func_pin(n,cth)*spherical_hn(n,z)/z
    N2 = np.cos(phi)*func_taon(n, cth)*(z*spherical_hn(n-1, z) - n*spherical_hn(n, z))/z
    N3 = -np.sin(phi)*func_pin(n, cth)*(z*spherical_hn(n-1, z) - n*spherical_hn(n, z))/z
    return N1, N2, N3


def calc_efield(r, theta, phi, m, x, n_max):  #define Ei and then sum all the i # use peter's formula.
    esum1 = 0
    esum2 = 0
    esum3 = 0
    for n in range(1, n_max+1):
        a_i = a_n(m, n, x)      # unitless
        b_i = b_n(m, n, x)
        [M1, M2, M3] = M_3oln(r, theta, phi, 1, x/radius_input, n)  # unitless
        [N1, N2, N3] = N_3eln(r, theta, phi, 1, x/radius_input, n)
        f_n = (1j)**n*(2*n+1)/(n*(n+1))    # unitless. 
        E_i1 = f_n*(1j*a_i*N1 - b_i*M1)
        E_i2 = f_n*(1j*a_i*N2 - b_i*M2)
        E_i3 = f_n*(1j*a_i*N3 - b_i*M3)
        esum1 += E_i1
        esum2 += E_i2
        esum3 += E_i3
    return np.linalg.norm([esum1, esum2, esum3])

st.header('Material Propeties')

# a material to be chosen 
material = st.selectbox(
    'What material do you want to calculate?',
    ('Ag(Drude)', 'Al(Drude)'))
st.write(f'You selected: {material}')

#TODO add more materials such as Cupper and Platinum

# the radius
radius_input = st.number_input('Input the radius', value=50.0, step=0.1)
st.write(f'The current radius is {radius_input} nm')

# the radius
n_medium = st.number_input('Input the refractive index of the surrounding medium', value=1.00)
st.write(f'The current refractive index of the surrounding medium is {n_medium}')

st.header('Mie Calculation Result')

n_max = 10
eps_inf = 5
w_p = 9.5
gamma = 0.1

wavelength = np.linspace(200, 800, 100)
omega = 1240 / wavelength
eps = eps_inf - w_p ** 2 / (omega * (omega + 1j * gamma))
n = np.sqrt(eps)
m = n / n_medium
x = 2 * pi / wavelength * radius_input

# mie calculation
c_sca, c_ext = mie(m, x, n_max, radius_input)
c_abs = c_ext - c_sca
calc_results_df = pd.DataFrame({'wavelength': wavelength, 'Cext': c_ext, 'Csca': c_sca, 'Cabs': c_abs})

# plot results in plotly
fig = px.line(calc_results_df, x='wavelength', y=calc_results_df.columns[0:4], 
                labels={
                    "variable": "cross-sections",
                    "value": "Cross-sections (nm^2)",
                    "wavelength": "Wavelength (nm)"}, 
                color_discrete_sequence=['red', 'blue', 'green'],
                title=f'{material}, Radius: {radius_input} nm, Refractive index of medium: {n_medium}'
                )
st.plotly_chart(fig, use_container_width=True)

# e-field plot
wavelength_efield = st.number_input('Input Wavelength (nm)', 600)
omega_efield = 1240 / wavelength_efield
eps = eps_inf - w_p ** 2 / (omega_efield * (omega_efield + 1j * gamma))
n = np.sqrt(eps)
m = n / n_medium
x = 2 * pi / wavelength_efield * radius_input

polar_p = np.linspace(0, 2*pi, 100)   # nm
polar_t = np.linspace(0, pi, 50)   # nm
pphi, ptheta = np.meshgrid(polar_p, polar_t)

zz = np.zeros([len(polar_t), len(polar_p)])
for i in range(len(polar_p)):
    phi = polar_p[i]
    for j in range(len(polar_t)):
        theta = polar_t[j]
        zz[j, i] = calc_efield(radius_input, theta, phi, m, x, n_max)

X = radius_input * np.sin(ptheta) * np.cos(pphi)
Y = radius_input * np.sin(ptheta) * np.sin(pphi)
Z = radius_input * np.cos(ptheta)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(1, 1, 1, projection='3d')
plot = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('jet'), facecolors=cm.jet(zz/np.max(zz)), linewidth=0, antialiased=False, alpha=0.5)
m = cm.ScalarMappable(cmap=cm.jet)
m.set_array(zz)
plt.colorbar(m)
plt.title(f'Field Enhancement of {material}(r={radius_input}nm) at {wavelength_efield} nm')
st.pyplot(fig)

# plot results in plotly
# fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z)])
# fig.update_layout(title=f'Field Enhancement of {material}(r={radius_input}nm) at {wavelength_efield} nm', autosize=False,
#                   width=500, height=500,
#                   margin=dict(l=65, r=50, b=65, t=90))
# st.plotly_chart(fig, use_container_width=True)

# @st.cache
# def convert_df(df):
#     # IMPORTANT: Cache the conversion to prevent computation on every rerun
#     return df.to_csv().encode('utf-8')

# csv = convert_df(calc_results_df)

# st.download_button(
#     label="Download data as CSV",
#     data=csv,
#     file_name='mie_calulation_results.csv',
#     mime='text/csv',
# )