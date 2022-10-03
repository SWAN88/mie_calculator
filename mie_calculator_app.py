import pandas as pd
import numpy as np
import streamlit as st
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from cmath import pi
from scipy.special import *
from bokeh.plotting import figure
import plotly.express as px

"""

# :full_moon: :chart_with_upwards_trend: Mie Calculator

Here you can calculate the optical cross-sections of spherical nanoparticles using Mie theory :heart:.

***Instructions***

Select a material from the dropdown menu and enter the radius in nanometers. 

Enter the refractive index of the media surrounding the particle, and enter the starting and ending wavelengths of the spectral range to be calculated.

Press “Plot!” to calculate the extinction, absorption and scattering cross sections for a single particle.

A csv file of the data can be downloaded; note that the cross-section units in the downloaded data are in units of m2/particle.

"""

st.header('Material Propeties')

# a material to be chosen 

option = st.selectbox(
    'What material do you want to calculate?',
    ('Silver', 'Aluminum'))
st.write(f'You selected: {option}')

# option = st.selectbox(
#     'What material do you want to calculate?',
#     ('Gold', 'Silver', 'Cupper', 'Alminum', 'Platinum'))
# st.write(f'You selected: {option}')

# the radius
radius_input = st.number_input('Input the radius', value=50.0, step=0.1)
st.write(f'The current radius is {radius_input} nm')

# the radius
n_medium = st.number_input('Input the refractive index of the surrounding medium', value=1.00)
st.write(f'The current refractive index of the surrounding medium is {n_medium}')

# the wavelength range
wavelength_start = st.number_input('Input the starting wavelength range', value=50, step=1)
st.write(f'The current start is {wavelength_start} nm')

wavelength_end = st.number_input('Input the end of wavelength range', value=1200, step=1)
st.write(f'The current end is {wavelength_end} nm')

wavelength_range = np.array(range(int(wavelength_start), int(wavelength_end), 1)) * 1e-9

# Make sparse output arrays
wavelength, radius = np.meshgrid(wavelength_range, radius_input * 1e-9)

# constants
eV = 6.582e-16  # hbar (eV*s)
k = 2 * pi / wavelength  # wavenumber
w = 3e8 * k  # frequency

# plot dielectric function
plot_epsilon = st.checkbox('Plot the dielectric function')

def plot_epslion_func(wavelength, eps):
    fig, ax = plt.subplots()
    ax.plot(wavelength[0] * 1e9, eps.real.T, label='Real', linewidth=2)
    ax.plot(wavelength[0] * 1e9, eps.imag.T, label='Imaginary', linewidth=2)
    ax.legend(fontsize=15)
    ax.set_xlabel('Wavelength (nm)', fontsize=20)
    ax.set_ylabel('Refractive index', fontsize=20)
    st.pyplot(fig)


def calc_epsilon(eps_inf, wb, tau):
    eps = eps_inf - wb ** 2 / (w * (w + 1j * tau))
    n = np.sqrt(eps)
    wavelength_out = np.sqrt(2 + eps_inf) * 2 * pi * 3e8 / wb
    return eps, n, wavelength_out


if plot_epsilon:
    reference = 'The CRC Handbook of Chemistry and Physics'
    st.write(f"These dielectric values are taken from '{reference}'")

    if option=='Silver':  # Ag dielectric function 
        eps_inf = 5  # dielectric function at infinite frequency
        wb = 9.5 / eV
        tau = 0.1 / eV  # relaxation time
        eps, n, _ = calc_epsilon(eps_inf, wb, tau)
        plot_epslion_func(wavelength, eps)

    elif option=='Aluminum':
        eps_inf = 1.25  # dielectric function at infinite frequency
        wb = 15.8 / eV
        tau = 0.2 / eV  # relaxation time
        eps, n, _ = calc_epsilon(eps_inf, wb, tau)
        plot_epslion_func(wavelength, eps)

st.header('Mie Calculation Result')

eps_inf = 5  # dielectric function at infinite frequency
wb = 9.5 / eV
tau = 0.1 / eV  # relaxation time
eps, n, _ = calc_epsilon(eps_inf, wb, tau)

n_max = st.number_input('Input the number of iteration', value=30, step=1)
st.write(f'The current number of iteration is {n_max}')

n_max = int(n_max)

# my_bar = st.progress(0)
# for percent_complete in range(100):
#     time.sleep(0.1)
#     my_bar.progress(percent_complete + 1)

m = n / n_medium

x = k * radius  # the size parameter
C_sca = 0
C_ext = 0
C_sca_old = 0
C_ext_old = 0
er_ext = np.zeros((n_max + 1))
er_sca = np.zeros((n_max + 1))

for it in tqdm(range(n_max)):
    # Spherical bessel functions of the first kind of x
    s_bes_jx = spherical_jn(it + 1, x)
    s_bes_jx_old = spherical_jn(it, x)
    # Spherical bessel functions of the first kind of mx
    s_bes_jmx = spherical_jn(it + 1, m * x)
    s_bes_jmx_old = spherical_jn(it, m * x)
    # Spherical bessel functions of the third kind
    s_bes_hx = spherical_jn(it + 1, x) + 1j * spherical_yn(it + 1, x)
    s_bes_hx_old = spherical_jn(it, x) + 1j * spherical_yn(it, x)

    # The derivatives of each spherical bessel function
    xj_prime = x * s_bes_jx_old - (it + 1) * s_bes_jx
    mxj_prime = m * x * s_bes_jmx_old - (it + 1) * s_bes_jmx
    xh_prime = x * s_bes_hx_old - (it + 1) * s_bes_hx

    # Mie scattering coefficients
    an = (m ** 2 * s_bes_jmx * xj_prime - s_bes_jx * mxj_prime) / (m ** 2 * s_bes_jmx * xh_prime - s_bes_hx * mxj_prime)
    bn = (s_bes_jmx * xj_prime - s_bes_jx * mxj_prime) / (s_bes_jmx * xh_prime - s_bes_hx * mxj_prime)
    # cn = (s_bes_jx * xh_prime - s_bes_hx * xj_prime) / (s_bes_jmx * xh_prime - s_bes_hx * mxj_prime)
    # dn = (m * s_bes_jx * xh_prime - m_ag * s_bes_hx * xj_prime) / (m ** 2 * s_bes_jmx * xh_prime - s_bes_hx * mxj_prime)

    # Cross section calculations
    C_sca = C_sca + (2 * (it + 1) + 1) * (np.abs(an) ** 2 + np.abs(bn) ** 2) * 2 * pi / k ** 2
    C_ext = C_ext + (2 * (it + 1) + 1) * (an + bn).real * 2 * pi / k ** 2

    # errors
    er_sca[it] = np.sum(np.sum(np.abs(C_sca - C_sca_old) ** 2 / np.abs(C_sca) ** 2))
    er_ext[it] = np.sum(np.sum(np.abs(C_ext - C_ext_old) ** 2 / np.abs(C_ext) ** 2))

    C_ext_old = C_ext
    C_sca_old = C_sca

# Calculating C_abs
C_abs = C_ext - C_sca

# calculation results in dataframe
wavelength_df = pd.DataFrame(wavelength[0]*1e9, columns=['wavelength'])
C_sca_df = pd.DataFrame(C_sca[0], columns=['Csca'])
C_abs_df = pd.DataFrame(C_abs[0], columns=['Cabs'])
C_ext_df = pd.DataFrame(C_ext[0], columns=['Cext'])
calc_results_df = pd.concat([wavelength_df, C_sca_df, C_abs_df, C_ext_df], axis=1)

st.subheader('Plotly')
# plot results in plotly
fig = px.line(calc_results_df, x='wavelength', y=calc_results_df.columns[0:4], 
                labels={
                    "variable": "cross-sections",
                    "value": "Cross-sections (nm^2)",
                    "wavelength": "Wavelength (nm)"}, 
                color_discrete_sequence=['red', 'blue', 'green'],
                title=f'{option}, Radius: {radius_input} nm, Refractive index of medium: {n_medium}')
st.plotly_chart(fig, use_container_width=True)

st.subheader('Bokeh')
# plot results in Bokeh
p = figure(
    title=f'{option}, Radius: {radius_input} nm, Refractive index of medium: {n_medium}',
    x_axis_label='Wavelength (nm)',
    y_axis_label='Cross section (nm^2)')

p.line(calc_results_df['wavelength'], calc_results_df['Csca'], legend_label='Csca', line_color='red', line_width=2)
p.line(calc_results_df['wavelength'], calc_results_df['Cabs'], legend_label='Cabs', line_color='blue', line_width=2)
p.line(calc_results_df['wavelength'], calc_results_df['Cext'], legend_label='Cext', line_color='green', line_width=2)
st.bokeh_chart(p, use_container_width=True)

st.subheader('Matplotlib')
# plot results in matplotlib
fig, ax = plt.subplots()
ax.plot(calc_results_df['wavelength'], calc_results_df['Csca'], label='Csca', c='r')
ax.plot(calc_results_df['wavelength'], calc_results_df['Cabs'], label='Cabs', c='b')
ax.plot(calc_results_df['wavelength'], calc_results_df['Cext'], label='Cext', c='g')
ax.set_title(f'{option}, Radius: {radius_input} nm, Refractive index of medium: {n_medium}')
ax.legend(fontsize=15)
ax.set_xlabel('Wavelength (nm)', fontsize=20)
ax.set_ylabel('$Cross-section (nm^2)$', fontsize=20)
st.pyplot(fig)

st.subheader('Error calculations')

# plot error calculations
plot_errors = st.checkbox('Plot error calculations')

if plot_errors:
    fig, ax = plt.subplots()
    ax.plot(range(n_max + 1), er_sca[:], label='Csca', c='r')
    ax.plot(range(n_max + 1), er_ext[:], label='Cext', c='g')
    ax.set_title(f'Number of iterations: {n_max}')
    ax.legend(fontsize=15)
    ax.set_xlabel('Number of iterations', fontsize=20)
    ax.set_ylabel('Errors', fontsize=20)
    st.pyplot(fig)

# st.subheader("Constants :globe_with_meridians:")

# st.latex(r'''
# h = 4.1357 \times 10^{-15} \space (eV \cdot s), 
# c = 2.9979 \times 10^{17} \space (nm/s),
# hc = 1239.8415 \space (eV \cdot nm),
# ''')

st.subheader("Dielectric functions :black_nib:")

st.latex(r'''
E (eV) = \frac{hc}{\lambda (nm)},
\nu (THz)= \frac{c}{\lambda (nm)} \times 10^{15},
T(fs) = \frac{1}{\nu(THz)} \times 10^{3}
''')