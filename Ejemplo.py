import rho_winds as rw
import astropy.units as u
import astropy.constants as const
import matplotlib.pyplot as plt

# ---------------------------
# Parámetros del ejemplo
# ---------------------------
M_HD = 0.99 * u.M_sun
r_HD = 0.912 * u.R_sun
T_HD = 4.27 * u.MK
rho_0 = 3.5e7 * const.m_p.to(u.kg) / u.cm**3
gamma = 1.05

# ---------------------------
# Viento politrópico
# ---------------------------
r_poly, v_poly, T_poly, a_poly = rw.viento_politropico(M_HD, r_HD, T_HD, gamma)

m_punto_poly = rw.perd_masa(r_HD, T_HD, rho_0, M_HD, gamma)
rho_poly = rw.densidad(T_HD, T_poly, rho_0, gamma) * 1000  # kg/m^3

# ---------------------------
# Preparar vectores para plot
# ---------------------------
r_vals = r_poly.to(u.R_sun).value
v_vals_km_s = v_poly.to(u.km/u.s).value
T_vals_MK = T_poly.to(u.MK).value
rho_plot = rho_poly.to(u.kg/u.m**3).value
mach_number = (v_poly / a_poly).decompose().value

# ---------------------------
# Plots viento politrópico
# ---------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
ax1, ax2, ax3, ax4 = axes.flatten()

ax1.semilogy(r_vals, rho_plot, color='C0')
ax1.set_xlabel('r [R_sun]')
ax1.set_ylabel(r'$\rho$ [kg m$^{-3}$]')
ax1.set_title('Density')
ax1.grid(True, which='both', ls='--', lw=0.5)

ax2.plot(r_vals, v_vals_km_s, color='C1')
ax2.set_xlabel('r [R_sun]')
ax2.set_ylabel('v [km/s]')
ax2.set_title('Velocity')
ax2.grid(True, ls='--', lw=0.5)

ax3.plot(r_vals, T_vals_MK, color='C2')
ax3.set_xlabel('r [R_sun]')
ax3.set_ylabel('T [MK]')
ax3.set_title('Temperature')
ax3.grid(True, ls='--', lw=0.5)

ax4.plot(r_vals, mach_number, color='C3')
ax4.set_xlabel('r [R_sun]')
ax4.set_ylabel('v / a')
ax4.set_title('Mach number (v/a)')
ax4.grid(True, ls='--', lw=0.5)

plt.tight_layout()
mdot_val = m_punto_poly.to(u.M_sun / u.yr).value
fig.suptitle(f'Mass loss: M_dot = {mdot_val:.2e} M_sun/yr', y=1.02, fontsize=12)

plt.savefig('Ejemplo_poly.png')
plt.show()

# ---------------------------
# Viento de Parker
# ---------------------------
r_parker, v_parker, r_c_parker, a_parker = rw.parker(M_HD, r_HD, T_HD)
mdot_parker = rw.perd_masa_parker(r_parker, rho_0, v_parker)
rho_parker = rw.densidad_parker(r_parker, mdot_parker, v_parker)

r_vals_p = r_parker.to(u.R_sun).value
v_vals_p = v_parker.to(u.km/u.s).value
rho_plot_p = rho_parker.to(u.kg/u.m**3).value
mach_parker = (v_parker / a_parker).decompose().value

fig2, axes2 = plt.subplots(3, 1, figsize=(12, 8))

ax1p, ax2p, ax3p = axes2.flatten()

ax1p.semilogy(r_vals_p, rho_plot_p, color='C0')
ax1p.set_xlabel('r [R_sun]')
ax1p.set_ylabel(r'$\rho$ [kg m$^{-3}$]')
ax1p.set_title('Density (Parker)')
ax1p.grid(True, which='both', ls='--', lw=0.5)

ax2p.plot(r_vals_p, v_vals_p, color='C1')
ax2p.set_xlabel('r [R_sun]')
ax2p.set_ylabel('v [km/s]')
ax2p.set_title('Velocity (Parker)')
ax2p.grid(True, ls='--', lw=0.5)


ax3p.plot(r_vals_p, mach_parker, color='C3')
ax3p.set_xlabel('r [R_sun]')
ax3p.set_ylabel('v / a')
ax3p.set_title('Mach number (v/a) (Parker)')
ax3p.grid(True, ls='--', lw=0.5)

plt.tight_layout()
mdot_parker_val = mdot_parker.to(u.M_sun / u.yr).value
fig2.suptitle(f'Parker wind — Mass loss: M_dot = {mdot_parker_val:.2e} M_sun/yr', y=1.02, fontsize=12)

plt.savefig('Ejemplo_Parker.png')
plt.show()
