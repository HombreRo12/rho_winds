# %%
###################################################################################################################################################
#Importar librerías
import numpy as np
from scipy.integrate import solve_ivp
import astropy.units as u
import astropy.constants as const
import os
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar
###################################################################################################################################################
#Cálculos del viento estelar politrópico

# Función para encontrar el índice del valor más cercano en un array
def find_nearest_index(array, value):
	return (np.abs(array - value)).argmin()


# Constantes, por default en MKS
G_Pol = const.G.value
k_B_Pol = const.k_B.value
m_p_Pol	= const.m_p.value

# Velocidad del sonido al cuadrado, ahora hago una función porque no es constante
def a_cuad(T, gamma, mu):
 '''
 Velocidad del sonido al cuadrado para gas politrópico
 T: temperatura [K]
 gamma: índice politrópico
 mu: peso molecular medio
 Retorna: a^2 [m^2/s^2]
 '''
 return gamma * k_B_Pol * T / (mu * m_p_Pol)


# Ecuación diferencial del viento politrópico
def politropic_eq(r, y, M, gamma, T0, r0, mu):
    '''
    # Ecuaciones diferenciales del viento politrópico
    r: radio [m]
    y: vector [v, T]
    M: masa estelar [kg]
    gamma: índice politrópico
    T0: temperatura base
    r0: radio base
    mu: peso molecular
    Retorna: [dv/dr, dT/dr]
    '''
    v, T = y
    a2 = a_cuad(T, gamma, mu)

    dvdr_num = 2 * gamma * a2 / r - G_Pol * M / r**2
    dvdr_den = v**2 - gamma * a2
    dvdr = (v * dvdr_num) / dvdr_den

    # T(ρ) pero pongo ρ(v,r) así no tengo variables de más ni calculo ρ todavía.
    dlnTdr = (1 - gamma) * (2 / r + 1/ v * dvdr)
    dTdr = T * dlnTdr

    return [dvdr, dTdr]


def mismatch_Tc(Tc_guess, M, R, T0, gamma, mu):
    '''
    Función para encontrar la temperatura crítica Tc
    Tc_guess: valor inicial de Tc [K]
    M: masa estelar
    R: radio estelar
    T0: temperatura base
    gamma: índice politrópico
    mu: peso molecular
    Retorna: diferencia T(R) - T0 (para root_scalar)
    '''
    a2 = a_cuad(Tc_guess, gamma, mu)
    rc = G_Pol * M / (2 * gamma * a2)
    vc = np.sqrt(gamma * a2)

    delta = 1e-6
    v_in = vc * (1 - delta)

    sol = solve_ivp(politropic_eq, [rc * (1 - delta), R],
                    [v_in, Tc_guess],
                    args=(M, gamma, Tc_guess, R, mu),
                    rtol=2.3e-14)

    T_at_R = sol.y[1, -1]
    return T_at_R - T0


def viento_politropico(M, R, T0, gamma=1.05, mu=0.612, N=5000, max_range=1*u.AU):
    """
    Integra el viento politrópico.
    
    Parámetros
    ----------
    M : astropy.units.Quantity
        Masa estelar.
    R : astropy.units.Quantity
        Radio estelar.
    T0 : astropy.units.Quantity
        Temperatura en la base (r = R).
    gamma : float
        Índice politrópico (default 1.05).
    mu : float, opcional
        Peso molecular medio (default 0.612).
    N : int, opcional
        Número total de puntos de integración (default 5000).
    max_range : astropy.units.Quantity, opcional
        Distancia máxima hasta la que se integra (default 1 AU).
    Retorna
    -------
    r_all : astropy.units.Quantity
        Distancia [R_sun].
    v_all : astropy.units.Quantity
        Velocidad [km/s].
    T_all : astropy.units.Quantity
        Temperatura [MK].
    a_all : astropy.units.Quantity
        Velocidad del sonido [km/s].
    """
    M_val = M.to(u.kg).value
    R_val = R.to(u.m).value
    T0_val = T0.to(u.K).value

    # Resolver Tc (temperatura crítica)
    sol_Tc = root_scalar(mismatch_Tc, bracket=[0.01*T0_val, 2*T0_val],
                         args=(M_val, R_val, T0_val, gamma, mu),
                         xtol=1e-14)
    Tc = sol_Tc.root

    # Punto crítico
    a2 = a_cuad(Tc, gamma, mu)
    rc = G_Pol * M_val / (2 * gamma * a2)
    vc = np.sqrt(gamma * a2)

    delta = 1e-6
    v_in = vc * (1 - delta)
    v_out = vc * (1 + delta)

    r_min = R_val
    r_max = max_range.to(u.m).value

    # Grillas de integración (N puntos totales)
    # Para integrar hacia adentro queremos t_eval que vaya desde rc*(1-delta) hacia r_min
    r_in_grid_decr  = np.linspace(rc * (1 - delta), r_min, N//2)  # decreciente en valor de r
    r_out_grid      = np.linspace(rc * (1 + delta), r_max, N//2)  # creciente

    # Integro hacia adentro: ***iniciar en el punto crítico*** y avanzar hacia r_min
    sol_in = solve_ivp(
        politropic_eq,
        [rc * (1 - delta), r_min],                # t_span comienza en rc*(1-delta)
        [v_in, Tc],                               # condiciones en el punto crítico
        t_eval=r_in_grid_decr,                    # grilla decreciente (rc -> r_min)
        args=(M_val, gamma, Tc, R_val, mu),
        rtol=2.3e-14
    )

    # Integro hacia afuera (desde el punto crítico hasta 1 AU)
    sol_out = solve_ivp(
        politropic_eq,
        [rc * (1 + delta), r_max],
        [v_out, Tc],
        t_eval=r_out_grid,
        args=(M_val, gamma, Tc, R_val, mu),
        rtol=2.3e-14
    )

    # sol_in.t vendrá en orden rc -> r_min (decreciente). Invertimos para obtener r_min -> rc.
    r_all = np.concatenate((sol_in.t[::-1], sol_out.t))
    v_all = np.concatenate((sol_in.y[0][::-1], sol_out.y[0]))
    T_all = np.concatenate((sol_in.y[1][::-1], sol_out.y[1]))
    a_all = np.sqrt(a_cuad(T_all, gamma, mu))

    return (r_all * u.m).to(u.R_sun), (v_all * u.m/u.s).to(u.km/u.s), (T_all * u.K).to(u.MK), (a_all * u.m/u.s).to(u.km/u.s)



def perd_masa(r_est, T0, rho, M_est, gamma=1.05, mu=0.612):
    '''   
 Cálculo de la tasa de pérdida de masa para viento politrópico
 r_est: radio base
 T0: temperatura base
 rho: densidad base
 M_est: masa estelar
 gamma: índice politrópico
 mu: peso molecular
 Retorna: pérdida de masa Mdot [M_sun/yr]
 '''

    M = M_est.to(u.kg).value
    r = r_est.to(u.m).value
    T0 = T0.to(u.K).value
    rho = rho.to(u.kg/u.m**3).value

    Rg = k_B_Pol / m_p_Pol
    v_esc = np.sqrt(2 * G_Pol * M / r)

    perd = 4 * np.pi * r**2 * rho * np.sqrt((gamma * Rg * T0)/mu) \
        * ((v_esc**2 / (4 * gamma * Rg * T0 / mu))**((gamma+1)/(2*(gamma-1)))) \
        * ((((8 * gamma * Rg * T0 / mu) / v_esc**2) - 4 * gamma + 4)/(5 - 3 * gamma))**((5 - 3 * gamma)/(2 * (gamma - 1)))

    perd = perd * u.kg / u.s
    perd = perd.to(u.M_sun / u.yr)
    return perd


def densidad(T0, T, rho0, gamma):
    '''
    Cálculo de la densidad a lo largo del viento politrópico
    T0: temperatura base
    T: vector de temperatura
    rho0: densidad base
    gamma: índice politrópico
    Retorna: vector de densidad
    '''
    densidad = rho0 * (T / T0)**(1/(gamma - 1))
    return densidad

###################################################################################################################################################
#Cálculo del viento de Parker
k_B = const.k_B.value
G = const.G.value
m_p	= const.m_p.value

def parker_eq(r, v, M, a):
    '''
    Ecuación diferencial para el viento de Parker
    r: radio
    v: velocidad
    M: masa estelar
    a: velocidad del sonido (isotérmico)
    Retorna: dv/dr
    '''
    ecu = v*(2 * a**2 / r - G * M / r**2) / (v**2 - a**2 )
    return ecu

def parker(M, R, T, mu=0.612, max_rang=1*u.AU):
    '''
    Función para integrar el viento de Parker (isotérmico)
    M, R, T: masa, radio y temperatura base
    mu: peso molecular
    max_rang: distancia máxima
    Retorna: r_combinado [R_sun], v_combinado [km/s], r_crit [R_sun], a [km/s]
    '''
    T = (T.to(u.K)).value
    R = (R.to(u.m)).value
    M = (M.to(u.kg)).value
    
    
    a = np.sqrt(k_B * T / (mu * m_p))  # Velocidad del sonido

    r_c = G * M / (2 * a**2)
    v_c = a

    # Radio en el que se resuelve la ecuación 
    r_min = R
    r_max = max_rang.to(u.m).value

    # Condiciones iniciales cerca del punto crítico (cerca para que no explote)
    delta = 1e-6
    v_izq = v_c * (1 - delta)
    v_der = v_c * (1 + delta)

    # Integrar hacia adentro
    sol_dentro = solve_ivp(parker_eq, [r_c * (1 - delta), r_min], [v_izq], args=(M, a), rtol=2.3e-14)

    # Integrar hacia afuera
    sol_afuera = solve_ivp(parker_eq, [r_c * (1 + delta), r_max], [v_der], args=(M, a), rtol=2.3e-14)

    # Juntar soluciones
    r_combinado = np.concatenate((sol_dentro.t[::-1], sol_afuera.t))
    v_combinado = np.concatenate((sol_dentro.y[0][::-1], sol_afuera.y[0]))
    
    # Unidades
    r_combinado = r_combinado * u.m
    v_combinado = v_combinado * u.m / u.s
    r_c	= r_c * u.m
    a = a * u.m / u.s

    return r_combinado.to(u.R_sun), v_combinado.to(u.km/u.s), r_c.to(u.R_sun), a.to(u.km/u.s)



def perd_masa_parker(r, rho_inicial, v):
    '''
    Pérdida de masa para viento de Parker
    r: vector de radios
    rho_inicial: densidad en la base
    v: velocidad en la base
    Retorna: Mdot [g/s]
    '''
    r_inicial	= r[0].to(u.cm)
    v_inicial	= v[0].to(u.cm/u.s)
    rho_inicial	= rho_inicial.to(u.g/u.cm**3)
    perd_m = 4 * np.pi * r_inicial**2 * rho_inicial * v_inicial
    return perd_m

def densidad_parker(r, perd_m, v):
    '''
    Densidad para viento de Parker
    r: vector de radios
    perd_m: pérdida de masa Mdot
    v: vector de velocidades
    Retorna: vector de densidades
    '''
    densidad = (perd_m / (4 * np.pi * r.to(u.cm)**2 * v.to(u.cm/u.s)))
    return densidad

