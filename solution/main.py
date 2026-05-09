import io
import os
import sys

import numpy as np
import skrf as rf
from scipy.optimize import differential_evolution, minimize, minimize_scalar

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "oracle"))
import oracle


def fetch_network(meas_name):
    resp = oracle.handle_query(meas_name, {})
    s2p_text = resp["s2p"]

    with io.StringIO(s2p_text) as buf:
        buf.name = f"{meas_name}.s2p"
        net = rf.Network(buf)

    return net

meas_db = {
    "measurement1": fetch_network("measurement1"),
    "measurement2": fetch_network("measurement2"),
    "measurement3": fetch_network("measurement3"),
    "measurement4": fetch_network("measurement4"),
}

freqs_hz = meas_db["measurement1"].f
ang_freq = 2 * np.pi * freqs_hz

eps_inf = 2.2
delta_eps = 1.5
tau_s = 100e-12
beta_frac = 0.9

c0 = 299792458.0
seg_len = 0.005
z_ref = 50.0

coarse_freq_idx = np.arange(0, len(freqs_hz), 8)


def cascade_chain(z_profile, alpha, idx=None):
    w = ang_freq if idx is None else ang_freq[idx]

    eps_eff = eps_inf + delta_eps / ((1 + (1j * w * tau_s) ** alpha) ** beta_frac)
    gamma = 1j * (w / c0) * np.sqrt(eps_eff)

    a_tot = np.ones_like(w, dtype=complex)
    b_tot = np.zeros_like(w, dtype=complex)
    c_tot = np.zeros_like(w, dtype=complex)
    d_tot = np.ones_like(w, dtype=complex)

    ch = np.cosh(gamma * seg_len)
    sh = np.sinh(gamma * seg_len)

    for z_seg in z_profile:
        a_seg = ch
        b_seg = z_seg * sh
        c_seg = (1.0 / z_seg) * sh
        d_seg = ch

        next_a = a_tot * a_seg + b_tot * c_seg
        next_b = a_tot * b_seg + b_tot * d_seg
        next_c = c_tot * a_seg + d_tot * c_seg
        next_d = c_tot * b_seg + d_tot * d_seg

        a_tot, b_tot, c_tot, d_tot = next_a, next_b, next_c, next_d

    denom = a_tot + b_tot / z_ref + c_tot * z_ref + d_tot
    s11 = (a_tot + b_tot / z_ref - c_tot * z_ref - d_tot) / denom
    s21 = 2.0 / denom
    s22 = (-a_tot + b_tot / z_ref - c_tot * z_ref + d_tot) / denom

    return s11, s21, s22


def make_z_profile(meas_name, m, b, phi, log_ztag, eta, n_start):
    ztag = np.exp(log_ztag)

    if meas_name == "measurement1":
        n_vals = np.arange(n_start, n_start + 100)
        z_line = 50.0 * np.exp(m * np.cos(2 * np.pi * b * n_vals + phi))

    elif meas_name == "measurement2":
        n_vals = np.arange(n_start + 1, n_start + 100)
        z_line = 50.0 * np.exp(m * np.cos(2 * np.pi * b * n_vals + phi))

    elif meas_name == "measurement3":
        n_vals = np.arange(n_start + 1, n_start + 100)
        tail = 50.0 * np.exp(m * np.cos(2 * np.pi * b * n_vals + phi))
        z_line = np.concatenate(([ztag], tail))

    elif meas_name == "measurement4":
        n_vals = np.arange(n_start, n_start + 100)
        z_line = 50.0 * np.exp(m * np.cos(2 * np.pi * b * n_vals + phi))
        z_line = z_line * np.exp(eta * n_vals)

    return z_line


def simulate_one_mode(meas_name, fit_params, n_start, idx=None):
    m, b, phi, alpha, log_ztag, eta = fit_params
    z_profile = make_z_profile(meas_name, m, b, phi, log_ztag, eta, n_start)
    return cascade_chain(z_profile, alpha, idx=idx)


def mode_fit_error(meas_name, fit_params, n_start, idx=None):
    s11_sim, s21_sim, s22_sim = simulate_one_mode(meas_name, fit_params, n_start, idx=idx)

    s_meas = meas_db[meas_name].s if idx is None else meas_db[meas_name].s[idx]
    s11_meas = s_meas[:, 0, 0]
    s21_meas = s_meas[:, 1, 0]
    s22_meas = s_meas[:, 1, 1]

    err_chunks = []
    err_chunks.append(np.mean(np.abs(s11_sim - s11_meas) ** 2))
    err_chunks.append(np.mean(np.abs(s21_sim - s21_meas) ** 2))
    err_chunks.append(np.mean(np.abs(s22_sim - s22_meas) ** 2))

    total_err = 0.0
    for piece in err_chunks:
        total_err += piece

    return total_err


def remove_eta(x5, eta=0.0):
    m, b, phi, alpha, log_ztag = x5
    return np.array([m, b, phi, alpha, log_ztag, eta], dtype=float)


def err_for_123_only(x5, n_start, idx=None):
    fit_params = remove_eta(x5, eta=0.0)
    running_err = 0.0
    for meas_name in ("measurement1", "measurement2", "measurement3"):
        running_err += mode_fit_error(meas_name, fit_params, n_start=n_start, idx=idx)
    return running_err


def err_for_all_modes(fit_params, which_modes, n_start, idx=None):
    total = 0.0
    for meas_name in which_modes:
        total += mode_fit_error(meas_name, fit_params, n_start=n_start, idx=idx)
    return total

eta_bounds = (-0.05, 0.05)

bounds_main3 = [
    (0.0, 1.5),
    (0.001, 0.499),
    (0.0, 2 * np.pi),
    (0.05, 1.0),
    (np.log(10.0), np.log(200.0)),
]


bounds_full = [
    (0.0, 1.5),
    (0.001, 0.499),
    (0.0, 2 * np.pi),
    (0.05, 1.0),
    (np.log(10.0), np.log(200.0)),
    eta_bounds,
]


def solve_first_three_modes(n_start, tries=5, de_iters=100, de_pop=15):
    best_x5 = None
    best_score = np.inf

    for attempt in range(tries):
        de_res = differential_evolution(lambda guess: err_for_123_only(guess, n_start=n_start, idx=coarse_freq_idx),
                                        bounds=bounds_main3, strategy="best1bin", maxiter=de_iters, popsize=de_pop,
                                        tol=0.01, mutation=(0.5, 1.0), recombination=0.7, polish=False, disp=False)

        local_res = minimize(lambda guess: err_for_123_only(guess, n_start=n_start, idx=None),
                             x0=de_res.x, method="L-BFGS-B", bounds=bounds_main3)

        guess_now = local_res.x.copy()
        guess_now[2] = guess_now[2] % (2 * np.pi)
        score_now = local_res.fun

        if score_now < best_score:
            best_score = score_now
            best_x5 = guess_now

        print(f"[123] run {attempt + 1}/{tries} | n_start={n_start} | obj={score_now:.6e}")

    return best_x5, best_score


def fit_eta_from_measurement4(x5, n_start):
    def eta_objective(eta_guess):
        fit_params = remove_eta(x5, eta=eta_guess)
        return mode_fit_error("measurement4", fit_params, n_start=n_start, idx=None)

    res = minimize_scalar(eta_objective, bounds=eta_bounds, method="bounded")
    return res.x, res.fun


def solve_all_four_modes(n_start, x6_seed=None, tries=5, de_iters=100, de_pop=15):
    best_x6 = None
    best_score = np.inf

    all_modes = ["measurement1", "measurement2", "measurement3", "measurement4"]

    for attempt in range(tries):
        de_res = differential_evolution(lambda guess: err_for_all_modes(guess,
                 which_modes=all_modes, n_start=n_start, idx=coarse_freq_idx),
                 bounds=bounds_full, strategy="best1bin", maxiter=de_iters,
                 popsize=de_pop, tol=0.01, mutation=(0.5, 1.0), recombination=0.7,
                 polish=False, disp=False)

        local_res = minimize(lambda guess: err_for_all_modes(guess, which_modes=all_modes, n_start=n_start, idx=None),
                             x0=de_res.x, method="L-BFGS-B", bounds=bounds_full)

        fit_now = local_res.x.copy()
        fit_now[2] = fit_now[2] % (2 * np.pi)
        score_now = local_res.fun

        if score_now < best_score:
            best_score = score_now
            best_x6 = fit_now

        print(f"ALL run {attempt + 1}/{tries}, n_start={n_start}, obj={score_now:.6e}")

    return best_x6, best_score


x5_start0, err123_start0 = solve_first_three_modes(n_start=0, tries=5)
x5_start1, err123_start1 = solve_first_three_modes(n_start=1, tries=5)

print("\nBest 123 fit with n_start=0")
print("  x5 =", x5_start0)
print("  fitted ztag =", np.exp(x5_start0[4]))
print("  obj123 =", err123_start0)

print("\nBest 123 fit with n_start=1")
print("  x5 =", x5_start1)
print("  fitted ztag =", np.exp(x5_start1[4]))
print("  obj123 =", err123_start1)

eta0, err4_start0 = fit_eta_from_measurement4(x5_start0, n_start=0)
eta1, err4_start1 = fit_eta_from_measurement4(x5_start1, n_start=1)

x6_start0 = remove_eta(x5_start0, eta=eta0)
x6_start1 = remove_eta(x5_start1, eta=eta1)

print("\nBest eta for measurement4 from start_index=0 fit:", eta0)
print("measurement4 prediction error from start_index=0 fit:", err4_start0)

print("\nBest eta for measurement4 from start_index=1 fit:", eta1)
print("measurement4 prediction error from start_index=1 fit:", err4_start1)

chosen_n_start = 0 if err4_start0 < err4_start1 else 1

print("\nChosen indexing convention:", f"n={chosen_n_start}..{chosen_n_start + 99}")

x6_final, err_all_final = solve_all_four_modes(n_start=chosen_n_start, tries=5)

m_final, b_final, phi_final, alpha_final, log_ztag_final, eta_final = x6_final
ztag_final = np.exp(log_ztag_final)

print("\nFinal fitted ztag:", ztag_final)
print("Final fitted eta:", eta_final)
print("Final all-mode objective:", err_all_final)
print(f"{m_final:.2f}, {b_final:.2f}, {phi_final:.2f}, {alpha_final:.2f}")
