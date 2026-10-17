import numpy as np
import pandas as pd

# ==============================
# AIP Test function
# ==============================
def check_AIP_and_case(m, M, kappa, mu_t):
    """
    Check if Arbitrage-free Interval Property (AIP) holds and determine which case applies.
    """
    N = len(mu_t)
    m_1 = m * (1 + mu_t[0])
    M_N = M * (1 + mu_t[-1])
    aip_threshold = max(m_1 - 1, 1 - M_N)
    AIP_holds = kappa >= aip_threshold

    if not AIP_holds:
        return False, None, aip_threshold

    # Case determination according to Corollary 4.21
    if kappa >= abs(m_1 - 1):
        return True, 1, aip_threshold
    elif (1 - M_N) <= kappa <= (1 - m_1):
        return True, 2, aip_threshold
    else:
        return True, 1, aip_threshold  # fallback to Case 1

# ==============================
# Payoff Parameters Case 1
# ==============================
import numpy as np

def compute_case1_payoff_params(m, M, kappa, mu_t, ghat_t, eps_small=1e-12):
    """
    Returns (new_mu_array, new_ghat_list) implementing Prop. Conv11 (Case 1) with safer numerics.
    - m, M: scalars (alpha_{t-1}, beta_{t-1})
    - kappa: transaction cost parameter
    - mu_t: array-like of length N (hat_mu_t^i)
    - ghat_t: list of N callables: hat_g_t^i(x)
    """
    N = len(mu_t)
    mu_t = np.asarray(mu_t)
    # alpha^i = alpha_{t-1} * (1 + mu_t[i])
    alpha_arr = m * (1 + mu_t)
    beta_arr  = M * (1 + mu_t)

    # tilde_g_{t-1}^i: i=0..N-1 -> ghat_t[i](alpha_{t-1} * x)
    #                i=N..2N-1 -> ghat_t[i-N](beta_{t-1} * x)
    def make_tilde(i):
        if i < N:
            def f(x, idx=i):
                return ghat_t[idx](m * x)   # alpha_{t-1} * x
        else:
            def f(x, idx=i-N):
                return ghat_t[idx](M * x)   # beta_{t-1} * x
        return f
    tilde_g_funcs = [make_tilde(i) for i in range(2 * N)]

    # rho = (1 + kappa) - (1 + mu_t[0]) * m  (theory)
    rho = (1.0 + kappa) - (1.0 + mu_t[0]) * m

    if rho < -1e-12:
        # theory expects rho >= 0; warn but continue (you may prefer to raise)
        # For numerical tolerance allow tiny negative due to floating point
        print(f"Warning: rho = {rho:.3e} < 0. Check precondition kappa >= |alpha_{t-1}^1 - 1|.")

    # compute w (eps) robustly using alpha/beta differences
    eps = np.empty(2 * N, dtype=float)
    eps.fill(np.nan)
    eps[0] = 1.0
    # i = 1..N-1
    for i in range(1, N):
        denom = alpha_arr[i] - alpha_arr[0]
        if abs(denom) < eps_small:
            eps[i] = 1.0 - rho / eps_small
        else:
            eps[i] = 1.0 - rho / denom
    # i = N..2N-1 (beta - alpha^1)
    for i in range(N, 2 * N):
        denom = beta_arr[i - N] - alpha_arr[0]
        if abs(denom) < eps_small:
            eps[i] = 1.0 - rho / eps_small
        else:
            eps[i] = 1.0 - rho / denom

    # index sets
    Lambda_2 = [j for j in range(2 * N) if eps[j] > 0]        # I_11 in paper
    Lambda_1 = [(i, j) for i in range(2 * N) for j in range(2 * N)
                if (eps[i] <= 0) and (eps[j] > 0)]            # I_12 in paper

    # compute lambda_{i,j} with numerical guard and clamp to [0,1]
    lambda_ij = {}
    for (i, j) in Lambda_1:
        denom = abs(eps[j] - eps[i])
        if denom < eps_small:
            lam = 0.0
        else:
            lam = abs(eps[j]) * (1.0 - eps[i]) / denom
        # clamp
        lam = max(0.0, min(1.0, lam))
        lambda_ij[(i, j)] = lam

    # compute hat_mu_{t-1}^{i,j} : equals kappa + w^i * rho / (1 - w^i) if j == index_of_first (0), else kappa
    mu_ij = {}
    for (i, j) in Lambda_1:
        if abs(1.0 - eps[i]) < eps_small:
            extra = 0.0
        else:
            extra = eps[i] * rho / (1.0 - eps[i])
        mu_val = kappa + extra
        # according to the paper the special value applies when j == 1 (i.e. index 0 here)
        mu_ij[(i, j)] = mu_val if (j == 0) else kappa

    # hat_mu_{t-1}^i for Lambda_2 (I_11)
    mu_1i = {}
    for i in Lambda_2:
        mu_1i[i] = (rho - kappa) if (i == 0) else kappa

    # Build resulting lists (order doesn't matter mathematically, but keep deterministic)
    new_mu = []
    new_ghat = []

    # Lambda_1: pairwise convex combos
    for (i, j) in Lambda_1:
        mu_val = mu_ij[(i, j)]
        lam = lambda_ij[(i, j)]
        def make_ghat_pair(lam_val, idx_i=i, idx_j=j):
            def g(x):
                return lam_val * tilde_g_funcs[idx_i](x) + (1.0 - lam_val) * tilde_g_funcs[idx_j](x)
            return g
        new_mu.append(mu_val)
        new_ghat.append(make_ghat_pair(lam))

    # Lambda_2: mixes with tilde_g^1 (index 0)
    for i in Lambda_2:
        mu_val = mu_1i[i]
        w = eps[i]
        def make_ghat_single(w_val, idx=i):
            def g(x):
                return w_val * tilde_g_funcs[0](x) + (1.0 - w_val) * tilde_g_funcs[idx](x)
            return g
        new_mu.append(mu_val)
        new_ghat.append(make_ghat_single(w))

    return np.array(new_mu), new_ghat

# ==============================
# Payoff Parameters Case 2
# ==============================
def compute_case2_payoff_params(m, M, kappa, mu_t, ghat_t):
    N = len(mu_t)
    m_arr = np.array([m * (1 + mu_t[i]) for i in range(N)])
    M_arr = np.array([M * (1 + mu_t[i]) for i in range(N)])

    # tilde_g functions
    def tilde_g(i, x):
        if i < N:
            return ghat_t[i](m * x)
        else:
            return ghat_t[i - N](M * x)
    tilde_g_funcs = [lambda x, idx=i: tilde_g(idx, x) for i in range(2*N)]

    rho = [(1 - (-1)**k * kappa) - m * (1 + mu_t[0]) for k in [1,2]]

    eps = np.zeros((2*N, 2))
    eps_small = 1e-12
    for k in [0, 1]:
        for i in range(1, N):
            denom = m * (mu_t[i] - mu_t[0])
            eps[i, k] = 1 - rho[k] / denom if denom != 0 else 1 - rho[k] / eps_small
        for i in range(N, 2*N):
            denom = M_arr[i - N] - m_arr[0]
            eps[i, k] = 1 - rho[k] / denom if denom != 0 else 1 - rho[k] / eps_small

    Lambda_3 = []
    Lambda_4 = []
    for i in range(1, 2*N):
        for j in range(1, 2*N):
            for k in [0, 1]:
                for m_idx in [0, 1]:
                    if eps[i, k] <= 0 and eps[j, m_idx] > 0:
                        Lambda_3.append((i, j, k, m_idx))
                    if eps[j, m_idx] > 0:
                        Lambda_4.append((j, m_idx))

    lambda_ijkm = {}
    for (i, j, k, m_idx) in Lambda_3:
        denom = abs(eps[j, m_idx] - eps[i, k])
        lambda_ijkm[(i, j, k, m_idx)] = abs(eps[j, m_idx]) * (1 - eps[i, k]) / denom if denom != 0 else 0

    mu_ijkm = {}
    for (i, j, k, m_idx) in Lambda_3:
        denom = abs(eps[j, m_idx] - eps[i, k])
        term1 = (-1)**k * abs(eps[j, m_idx])
        term2 = (-1)**m_idx * abs(eps[i, k])
        mu_ijkm[(i, j, k, m_idx)] = -kappa + (term1 + term2) / denom if denom != 0 else -kappa

    mu_1jm = {}
    for (j, m_idx) in Lambda_4:
        mu_1jm[(j, m_idx)] = (-1)**m_idx * kappa

    # Collect new parameters
    new_mu = []
    new_ghat = []

    # Lambda_3: g_{T-1}^{(i,j),(k,m)}
    for (i, j, k, m_idx) in Lambda_3:
        mu_val = mu_ijkm[(i, j, k, m_idx)]
        def ghat_func(x, i=i, j=j, k=k, m_idx=m_idx):
            l = lambda_ijkm[(i, j, k, m_idx)]
            return l * tilde_g_funcs[i](x) + (1 - l) * tilde_g_funcs[j](x)
        new_mu.append(mu_val)
        new_ghat.append(ghat_func)
    
    # Lambda_4: g_{T-1}^{1,j,m}
    for (j, m_idx) in Lambda_4:
        mu_val = mu_1jm[(j, m_idx)]
        def ghat_func(x, j=j, m_idx=m_idx):
            return eps[j, m_idx] * tilde_g_funcs[0](x) + (1 - eps[j, m_idx]) * tilde_g_funcs[j](x)
        new_mu.append(mu_val)
        new_ghat.append(ghat_func)

    return np.array(new_mu), new_ghat

# ==============================
# Clean mu and ghat
# ==============================
def clean_mu_ghat(mu, ghat):
    mu_array = np.array(mu)
    df = pd.DataFrame({"mu": mu_array, "idx": np.arange(len(mu_array))})
    new_mu = []
    new_ghat = []

    for val, group in df.groupby("mu"):
        funcs = [ghat[j] for j in group["idx"].values]
        if len(funcs) == 1:
            new_mu.append(val)
            new_ghat.append(funcs[0])
        else:
            def f_max(x, funcs=funcs):
                return max(f(x) for f in funcs)
            new_mu.append(val)
            new_ghat.append(f_max)
    # Sort after cleaning
    sorted_pairs = sorted(zip(new_mu, new_ghat), key=lambda p: p[0])
    new_mu, new_ghat = zip(*sorted_pairs)
    return np.array(new_mu), list(new_ghat)
# ==============================
# Case 1
def compute_case1_payoff_paramsoption2(alpha, beta, kappa, mu_t, ghat_T):
    """
    For Case 1: Compute new mu and payoff functions at T-1 for recursion.

    Returns:
        new_mu: list of floats (mu_{T-1}^*)
        new_ghat: list of callables (ghat_{T-1}^*)
    """
    N = len(mu_t)
    alpha_arr = np.array([alpha * (1 + mu_t[i]) for i in range(N)])
    beta_arr = np.array([beta * (1 + mu_t[i]) for i in range(N)])

    def tilde_g(i, x):
        if i < N:
            return ghat_T[i](alpha * x)
        else:
            return ghat_T[i-N](beta * x)
    tilde_g_funcs = [lambda x, idx=i: tilde_g(idx, x) for i in range(2*N)]

    rho = (1 + kappa) - (1 + mu_t[0]) * alpha

    eps = np.zeros(2*N)
    eps[0] = 1
    for i in range(1, N):
        denom = alpha * (mu_t[i] - mu_t[0])
        eps[i] = 1 - rho / denom if denom != 0 else 1
    for i in range(N, 2*N):
        denom = beta_arr[i-N] - alpha_arr[0]
        eps[i] = 1 - rho / denom if denom != 0 else 1

    Lambda_1 = [(i, j) for i in range(2*N) for j in range(2*N)
                if eps[i] <= 0 and eps[j] > 0]
    Lambda_2 = [j for j in range(2*N) if eps[j] > 0]

    lambda_ij = {}
    for (i, j) in Lambda_1:
        denom = abs(eps[j] - eps[i])
        lambda_ij[(i, j)] = abs(eps[j]) * (1 - eps[i]) / denom if denom != 0 else 0

    mu_ij = {}
    for (i, j) in Lambda_1:
        mu_val = kappa + (eps[i] * rho) / (1 - eps[i]) if (1 - eps[i]) != 0 else k
        mu_ij[(i, j)] = mu_val if j == 0 else k

    mu_1i = {}
    for i in Lambda_2:
        if i == 0:
            mu_1i[i] = rho - kappa
        else:
            mu_1i[i] = kappa

    # Collect new parameters
    new_mu = []
    new_ghat = []

    # Lambda_1: g_{T-1}^{i,j}
    for (i, j) in Lambda_1:
        mu_val = mu_ij[(i, j)]
        def ghat_func(x, i=i, j=j):
            l = lambda_ij[(i, j)]
            return l * tilde_g_funcs[i](x) + (1 - l) * tilde_g_funcs[j](x)
        new_mu.append(mu_val)
        new_ghat.append(ghat_func)
    
    # Lambda_2: g_{T-1}^{1,i}
    for i in Lambda_2:
        mu_val = mu_1i[i]
        def ghat_func(x, i=i):
            return eps[i] * tilde_g_funcs[0](x) + (1 - eps[i]) * tilde_g_funcs[i](x)
        new_mu.append(mu_val)
        new_ghat.append(ghat_func)

    return new_mu, new_ghat
# ==============================
# One step backward recursion
# ==============================
def one_step_backward_recursion(i, m, M, kappa, mu_t, ghat):
    AIP_holds, case, threshold = check_AIP_and_case(m, M, kappa, mu_t)
    if not AIP_holds:
        raise ValueError(f"AIP does not hold at step {i}.")
    if case == 1:
        new_mu, new_ghat = compute_case1_payoff_params(m, M, kappa, mu_t, ghat)
        print(f"   Step {i}: Case 1 applied.")
    elif case == 2:
        new_mu, new_ghat = compute_case2_payoff_params(m, M, kappa, mu_t, ghat)
        print(f"   Step {i}: Case 2 applied.")
    else:
        new_mu, new_ghat = compute_case1_payoff_params(m, M, kappa, mu_t, ghat)  # fallback
    new_mu, new_ghat = clean_mu_ghat(new_mu, new_ghat)
    return new_mu, new_ghat

# ==============================
# Evaluate ghat
# ==============================
def eval_ghat(ghat, x):
    return ghat(x)

# ==============================
# Optimal strategies Case1
# ==============================
def Optimal_strategies_Case1(G_0, V_0, S, m, M, kappa, all_mu, all_ghat, n, c, eps_small=1e-12):
    G = [G_0]
    V = [V_0]
    Theta_opt_starting_from_0 = [0]
    Theta_opt = []
    for i in range(n):
        mu_t = all_mu[i+1]
        ghat_t = all_ghat[i+1]
        N_t = len(mu_t)
        alpha_arr = np.array([m * (1 + mu_t[j]) for j in range(N_t)])
        beta_arr  = np.array([M * (1 + mu_t[j]) for j in range(N_t)])
        p_bar = np.full(2 * N_t - 1, np.nan)
        b_bar = np.full(2 * N_t - 1, np.nan)

        # first block: j = 2..N_t  (we map them to indices 0..N_t-2 in arrays)
        for jj in range(0, N_t - 1):
            denom = (alpha_arr[jj + 1] - alpha_arr[0]) * S[c, i]
            if abs(denom) < eps_small:
                continue
            p_bar[jj] = 1.0 / denom
            b_bar[jj] = p_bar[jj] * (
                eval_ghat(ghat_t[jj + 1], m * S[c, i])
                - eval_ghat(ghat_t[0], m * S[c, i])
            )

        # second block: j = N_t+1 .. 2N_t (map to indices N_t-1 .. 2N_t-2)
        # IMPORTANT: start at j_idx = N_t (python index) so (j_idx - N_t) starts at 0
        for jj in range(N_t, 2 * N_t - 1):
            k = jj - N_t  # 0 .. N_t-1
            denom = (beta_arr[k] - alpha_arr[0]) * S[c, i]
            if abs(denom) < eps_small:
                continue
            p_bar[jj - 1] = 1.0 / denom  # align into p_bar shape (see original shaping)
            b_bar[jj - 1] = p_bar[jj - 1] * (
                eval_ghat(ghat_t[k], M * S[c, i])
                - eval_ghat(ghat_t[0], m * S[c, i])
            )

        # Clean finite entries and sort by p
        valid = np.isfinite(p_bar) & np.isfinite(b_bar)
        if not np.any(valid):
            # nothing finite -> fallback
            p = np.array([])
            b = np.array([])
        else:
            p = p_bar[valid]
            b = b_bar[valid]
            sigma = np.argsort(p)
            p = p[sigma]
            b = b[sigma]

        # a_{t-1}(alpha) = max_j ( b_j - p_j * alpha )
        def a_t_minus_1(alpha):
            if p.size == 0:
                return -np.inf
            return np.max(b - p * alpha)

        # d and s arrays (Case 1: large transaction cost regime)
        # According to LaTeX algorithm: d^1 = φ_{t-2}(1-α^1)S_{t-1}, s^1 = 1
        # d^i = (1-α^1)S_{t-1}b^i, s^i = 1 - (1-α^1)S_{t-1}p^i for i=2,...,2N_t
        d = np.zeros(len(p))
        s = np.zeros(len(p))
        d[0] = (1 - alpha_arr[0]) * Theta_opt_starting_from_0[-1] * S[c, i]
        s[0] = 1                                                            
        for jj in range(1, len(p)):
            d[jj] = (1 - alpha_arr[0]) * S[c, i] * b[jj] 
            s[jj] = 1 - (1 - alpha_arr[0]) * S[c, i] * p[jj]

        def A(jj, alpha):
            return d[jj] + s[jj] * alpha

        def A_t_minus_1(alpha):
            return np.max([A(jj, alpha) for jj in range(len(p))])

        # Candidate I values
        I = []
        for jj in range(len(p)):
            for ll in range(len(p)):
                if jj != ll and s[jj] <= 0 and s[ll] >= 0:
                    if (s[ll] - s[jj]) != 0:
                        I_val = (d[jj] - d[ll]) / (s[ll] - s[jj])
                        if not any(abs(I_val - existing) < 1e-9 for existing in I):
                            I.append(I_val)

        def objective_function(alpha):
            return abs(A_t_minus_1(alpha) - G[-1])

        alpha_star = 0
        if I:
            alpha_candidates = [alpha for alpha in I if alpha >= 0]
            if alpha_candidates:
                alpha_star = min(alpha_candidates, key=objective_function)

        # Optimal strategy
        theta_prev = Theta_opt_starting_from_0[-1]
        if theta_prev < a_t_minus_1(alpha_star):  # condition de la prop.
            theta_opt = a_t_minus_1(alpha_star)
        else:
            theta_opt = theta_prev

        Theta_opt.append(theta_opt)
        Theta_opt_starting_from_0.append(theta_opt)

    return Theta_opt, Theta_opt_starting_from_0

# ==============================
# Optimal strategies Case2
# ==============================
def Optimal_strategies_Case2(G_0, V_0, S, m, M, kappa, all_mu, all_ghat, n, c):
    G = [G_0]
    V = [V_0]
    Theta_opt_starting_from_0 = [0]
    Theta_opt = []
    for i in range(n):
        mu_t = all_mu[i+1]
        ghat_t = all_ghat[i+1]
        N_t = len(mu_t)

        # Alpha and beta arrays
        alpha_arr = np.array([m * (1 + mu_t[j]) for j in range(N_t)])
        beta_arr = np.array([M * (1 + mu_t[j]) for j in range(N_t)])

        p_bar = np.zeros(2 * N_t - 1)
        b_bar = np.zeros(2 * N_t - 1)



        for j in range(N_t-1):
            p_bar[j] = 1 / ((alpha_arr[j + 1] - alpha_arr[0]) * S[c, i]) if (alpha_arr[j + 1] - alpha_arr[0]) != 0 else 0
            b_bar[j] = p_bar[j] * (
                eval_ghat(ghat_t[j + 1], m * S[c, i])
                - eval_ghat(ghat_t[0], m * S[c, i])
            )
        for j in range(N_t-1, 2 * N_t -1):
            p_bar[j] = 1 / ((beta_arr[j - N_t] - alpha_arr[0]) * S[c, i]) if (beta_arr[j - N_t] - alpha_arr[0]) != 0 else 0
            b_bar[j] = p_bar[j] * (
                eval_ghat(ghat_t[j - N_t], M * S[c, i])
                - eval_ghat(ghat_t[0], m * S[c, i])
            )

        # Sort p and b
        sigma = np.argsort(p_bar)
        p = p_bar[sigma]
        b = b_bar[sigma]
        p = p[np.isfinite(p)]
        b = b[np.isfinite(b)]

        # a_{t-1}(alpha)
        def a_t_minus_1(alpha):
            return np.max(b - p * alpha)

        # d and s arrays
        d = np.zeros((2, len(p)))
        s = np.zeros((2, len(p)))
        
        # Fix degenerate case when kappa=0 (both branches become identical)
        kappa_corrected = kappa if kappa != 0 else 1e-10  # tiny perturbation
        
        for ii in range(2):
            for jj in range(len(p)):
                d[ii, jj] = (
                    (1 - alpha_arr[0] - (-1) ** (ii + 1) * kappa_corrected) * S[c, i] * b[jj]
                    + (-1) ** (ii + 1) * kappa_corrected * Theta_opt_starting_from_0[-1] * S[c, i]
                )
                s[ii, jj] = 1 - (1 - alpha_arr[0] - (-1) ** (ii + 1) * kappa_corrected) * S[c, i] * p[jj]

        def A(ii, jj, alpha):
            return d[ii, jj] + s[ii, jj] * alpha

        def A_values_func(alpha):
            return [A(ii, jj, alpha) for ii in range(2) for jj in range(len(p))]

        def A_t_minus_1_2(alpha):
            return np.max(A_values_func(alpha))

        # Candidate I values
        I = []
        for ii in range(2):
            for kk in range(2):
                for jj in range(len(p)):
                    for ll in range(len(p)):
                        if (ii, jj) != (kk, ll) and s[ii, jj] <= 0 and s[kk, ll] >= 0:
                            if (s[kk, ll] - s[ii, jj]) != 0:
                                I_val = (d[ii, jj] - d[kk, ll]) / (s[kk, ll] - s[ii, jj])
                                if not any(abs(I_val - existing) < 1e-9 for existing in I):
                                    I.append(I_val)

        def objective_function(alpha):
            return abs(A_t_minus_1_2(alpha) - G[-1])

        alpha_star = 0
        alpha_candidates = []
        if I:
            alpha_candidates = [alpha for alpha in I if alpha >= 0]
            if alpha_candidates:
                alpha_star = min(alpha_candidates, key=objective_function)

        # Optimal strategy
        theta_opt = a_t_minus_1(alpha_star)
        
        Theta_opt.append(theta_opt)
        Theta_opt_starting_from_0.append(theta_opt)
    return Theta_opt, Theta_opt_starting_from_0

# ==============================
# Compute optimal strategy based on AIP
# ==============================
def compute_optimal_strategy(G0, V0, S, m, M, kappa, all_mu_LO, all_ghat_LO, n, c):
    AIP_holds, case_num, aip_threshold = check_AIP_and_case(m, M, kappa, all_mu_LO[1])
    if not AIP_holds:
        print(f"[Path {c}] AIP does not hold. kappa={kappa}, threshold={aip_threshold:.4f}")
        return None, None, case_num
    if case_num == 1:
        Theta_opt, Theta_opt_starting_from_0 = Optimal_strategies_Case1(
            G0, V0, S, m, M, kappa, all_mu_LO, all_ghat_LO, n, c
        )
        print(f"[Path {c}] Using Case 1 strategy")
    elif case_num == 2:
        Theta_opt, Theta_opt_starting_from_0 = Optimal_strategies_Case2(
            G0, V0, S, m, M, kappa, all_mu_LO, all_ghat_LO, n, c
        )
        print(f"[Path {c}] Using Case 2 strategy")
    return Theta_opt, Theta_opt_starting_from_0, case_num

# ==============================
# Portfolio value
# ==============================
def portfolio_value(S, c, n, kappa, Theta_opt, all_mu_LO, all_ghat_LO, V0, G0):
    G = [G0]
    V = [V0]
    for i in range(n):
        mu_t = all_mu_LO[i+1]
        ghat_t = all_ghat_LO[i+1]
        theta_i = Theta_opt[i]
        # Transaction cost
        cost = kappa * abs(theta_i - (Theta_opt[i-1] if i>0 else 0)) * S[c, i]
        # Portfolio update (check bounds)
        S_next = S[c, i+1] if i+1 < S.shape[1] else S[c, i]
        V_next = V[i] + theta_i * (S_next - S[c, i]) - cost
        V.append(V_next)
        G_candidates = [eval_ghat(f, S_next) - mu * theta_i * S_next for f, mu in zip(ghat_t, mu_t)]
        G.append(max(G_candidates))
    return V, G

# ==============================
# Super hedging error
# ==============================
def super_hedging_error(G, V, S, c):
    return (V[-1] - G[-1]) / S[c, -1]
#==============================================

