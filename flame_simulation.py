import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cantera as ct

os.makedirs('plots', exist_ok=True)

print(f"Cantera version: {ct.__version__}")

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'lines.linewidth': 2,
    'axes.grid': True,
    'grid.alpha': 0.35,
})

T_INITIAL    = 300.0
P_INITIAL    = ct.one_atm
PHI          = 1.0
MECHANISM    = 'gri30.yaml'
H2_FRACTIONS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

results = {
    'h2_frac'        : [],
    'flame_speed'    : [],
    'T_max'          : [],
    'NO_ppm'         : [],
    'OH_max'         : [],
    'H_max'          : [],
    'O_max'          : [],
    'grid_0'         : None,
    'T_profile_0'    : None,
    'HRR_profile_0'  : None,
    'grid_50'        : None,
    'T_profile_50'   : None,
    'HRR_profile_50' : None,
}

for x_H2 in H2_FRACTIONS:
    x_CH4 = 1.0 - x_H2
    label = f"H2={int(x_H2*100):d}%"
    print(f"\n{'-'*60}")
    print(f"  Solving case: {label}  (x_CH4={x_CH4:.1f}, x_H2={x_H2:.1f})")

    O2_stoich = x_CH4 * 2.0 + x_H2 * 0.5
    N2_per_O2 = 79.0 / 21.0
    O2_actual = O2_stoich / PHI
    N2_actual = O2_actual * N2_per_O2

    composition = (
        f"CH4:{x_CH4:.4f}, "
        f"H2:{x_H2:.4f}, "
        f"O2:{O2_actual:.6f}, "
        f"N2:{N2_actual:.6f}"
    )
    print(f"  Composition: {composition}")

    gas = ct.Solution(MECHANISM)
    gas.TPX = T_INITIAL, P_INITIAL, composition

    flame = ct.FreeFlame(gas, width=0.03)
    flame.set_refine_criteria(ratio=3, slope=0.06, curve=0.12, prune=0.01)
    flame.set_max_jac_age(50, 50)
    flame.set_time_step(1e-6, [2, 5, 10, 20])
    flame.solve(loglevel=0, auto=True)

    Su    = flame.velocity[0] * 100.0
    T_max = np.max(flame.T)

    idx_end = int(0.95 * len(flame.grid))
    NO_idx  = gas.species_index('NO')
    NO_ppm  = np.mean(flame.X[NO_idx, idx_end:]) * 1e6

    OH_idx = gas.species_index('OH')
    H_idx  = gas.species_index('H')
    O_idx  = gas.species_index('O')
    OH_max = np.max(flame.X[OH_idx, :])
    H_max  = np.max(flame.X[H_idx,  :])
    O_max  = np.max(flame.X[O_idx,  :])

    print(f"  Su       = {Su:.2f} cm/s")
    print(f"  T_max    = {T_max:.1f} K")
    print(f"  NO       = {NO_ppm:.2f} ppm")
    print(f"  OH_max   = {OH_max:.4e}")

    results['h2_frac'].append(x_H2 * 100)
    results['flame_speed'].append(Su)
    results['T_max'].append(T_max)
    results['NO_ppm'].append(NO_ppm)
    results['OH_max'].append(OH_max)
    results['H_max'].append(H_max)
    results['O_max'].append(O_max)

    grid_cm = flame.grid * 100.0
    HRR     = flame.heat_release_rate

    if x_H2 == 0.0:
        results['grid_0']        = grid_cm.copy()
        results['T_profile_0']   = flame.T.copy()
        results['HRR_profile_0'] = HRR.copy()
    elif x_H2 == 0.5:
        results['grid_50']        = grid_cm.copy()
        results['T_profile_50']   = flame.T.copy()
        results['HRR_profile_50'] = HRR.copy()

print("\n" + "="*60)
print("  All cases solved successfully.")
print("="*60)

h2_pct = results['h2_frac']

def finish_plot(fig, ax, xlabel, ylabel, title, fname):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fname, dpi=200)
    plt.close(fig)
    print(f"  Saved: {fname}")

fig, ax = plt.subplots()
ax.plot(h2_pct, results['flame_speed'], 'o-', color='royalblue',
        label='Laminar flame speed')
ax.set_xlim(-2, 52)
finish_plot(fig, ax,
            xlabel='H$_2$ fraction in fuel blend [vol.%]',
            ylabel='Laminar flame speed $S_u$ [cm/s]',
            title='Effect of H$_2$ enrichment on laminar flame speed',
            fname='plots/flame_speed.png')

fig, ax = plt.subplots()
ax.plot(h2_pct, results['T_max'], 's-', color='firebrick',
        label='Max. flame temperature')
ax.set_xlim(-2, 52)
finish_plot(fig, ax,
            xlabel='H$_2$ fraction in fuel blend [vol.%]',
            ylabel='Maximum flame temperature [K]',
            title='Effect of H$_2$ enrichment on maximum flame temperature',
            fname='plots/flame_temperature.png')

fig, ax = plt.subplots()
ax.plot(h2_pct, results['NO_ppm'], '^-', color='darkorange',
        label='NO emission')
ax.set_xlim(-2, 52)
finish_plot(fig, ax,
            xlabel='H$_2$ fraction in fuel blend [vol.%]',
            ylabel='NO mole fraction [ppm]',
            title='Effect of H$_2$ enrichment on NO emissions',
            fname='plots/NO_emission.png')

fig, ax = plt.subplots()
ax.plot(h2_pct, results['OH_max'], 'o-', color='green',  label='OH')
ax.plot(h2_pct, results['H_max'],  's-', color='purple', label='H')
ax.plot(h2_pct, results['O_max'],  '^-', color='teal',   label='O')
ax.set_xlim(-2, 52)
finish_plot(fig, ax,
            xlabel='H$_2$ fraction in fuel blend [vol.%]',
            ylabel='Peak mole fraction [-]',
            title='Effect of H$_2$ enrichment on key radical concentrations',
            fname='plots/radicals.png')

def shift_to_flame(grid, T):
    dT = np.gradient(T, grid)
    idx = np.argmax(dT)
    return grid - grid[idx]

fig, ax = plt.subplots()
g0_shifted  = shift_to_flame(results['grid_0'],  results['T_profile_0'])
g50_shifted = shift_to_flame(results['grid_50'], results['T_profile_50'])

ax.plot(g0_shifted,  results['T_profile_0'],  '-',  color='royalblue', label='0% H$_2$')
ax.plot(g50_shifted, results['T_profile_50'], '--', color='firebrick',  label='50% H$_2$')
ax.set_xlim(-0.5, 2.0)
finish_plot(fig, ax,
            xlabel='Position [cm]',
            ylabel='Temperature [K]',
            title='Spatial temperature profile: 0% vs 50% H$_2$',
            fname='plots/temperature_profile.png')

fig, ax = plt.subplots()
ax.plot(g0_shifted,  results['HRR_profile_0']  / 1e6, '-',  color='royalblue', label='0% H$_2$')
ax.plot(g50_shifted, results['HRR_profile_50'] / 1e6, '--', color='firebrick',  label='50% H$_2$')
ax.set_xlim(-0.5, 2.0)
finish_plot(fig, ax,
            xlabel='Position [cm]',
            ylabel='Heat release rate [MW/m$^3$]',
            title='Spatial heat release rate profile: 0% vs 50% H$_2$',
            fname='plots/HRR_profile.png')

print("\nAll plots saved successfully.")
