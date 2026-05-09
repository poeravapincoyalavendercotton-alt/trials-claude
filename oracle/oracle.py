import skrf as rf
import numpy as np
import os
import tempfile

def handle_query(mode: str, parameters: dict = None) -> dict:
    if mode == "help":
        return {"modes": ["measurement1", "measurement2", "measurement3", "measurement4"]}

    # HIDDEN TRUTHS
    m_true = 0.55
    b_true = 0.381966
    phi_true = 2.718
    alpha_true = 0.73

    freq = rf.Frequency(0.1, 40, 1001, 'ghz')
    omega = freq.w

    # Substrate Physics
    epsilon_inf = 2.2
    delta_epsilon = 1.5
    tau = 100e-12
    beta_frac = 0.9
    epsilon_cplx = epsilon_inf + (delta_epsilon) / ((1 + (1j * omega * tau)**alpha_true)**beta_frac)
    gamma = 1j * (omega / 299792458) * np.sqrt(epsilon_cplx)

    length_m = 0.005 # 5 mm per segment

    def build_chain(start_index: int, num_segments: int):
        chain = None
        for n in range(start_index, start_index + num_segments):
            Z_n = 50.0 * np.exp(m_true * np.cos(2 * np.pi * b_true * n + phi_true))
            media = rf.media.DefinedGammaZ0(freq, gamma=gamma, z0=Z_n)
            segment = media.line(length_m, unit='m')
            chain = segment if chain is None else chain ** segment
        return chain

    if mode == "measurement1":
        chain = build_chain(start_index=0, num_segments=100)

    elif mode == "measurement2":
        chain = build_chain(start_index=1, num_segments=99)

    elif mode == "measurement3":
        Z_tag = 75.0
        tag_media = rf.media.DefinedGammaZ0(freq, gamma=gamma, z0=Z_tag)
        tag_segment = tag_media.line(length_m, unit='m')
        tail = build_chain(start_index=1, num_segments=99)
        chain = tag_segment ** tail

    elif mode == "measurement4":
        chain = None
        eta = 0.01
        for n in range(0, 100):
            Z_n = 50.0 * np.exp(m_true * np.cos(2 * np.pi * b_true * n + phi_true))
            Z_n *= np.exp(eta * n)
            media = rf.media.DefinedGammaZ0(freq, gamma=gamma, z0=Z_n)
            segment = media.line(length_m, unit='m')
            chain = segment if chain is None else chain ** segment

    else:
        return {"error": f"Unknown mode: {mode}"}

    chain.add_noise_polar(mag_dev=0.002, phase_dev=0.5)
    fd, path = tempfile.mkstemp(suffix=".s2p")
    os.close(fd)
    chain.write_touchstone(path, r_ref=50)

    with open(path, "r") as f:
        ts = f.read()
    os.remove(path)
    return {"s2p": ts}
