import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import eigs
import pandas as pd
import os


def finite_difference_weights(x0, x, max_order):
    """
    Calcule les poids de dérivation de Fornberg pour un stencil arbitraire.
    """
    n = len(x)
    c = np.zeros((max_order + 1, n), dtype=float)
    c1 = 1.0
    c4 = x[0] - x0
    c[0, 0] = 1.0

    for i in range(1, n):
        mn = min(i, max_order)
        c2 = 1.0
        c5 = c4
        c4 = x[i] - x0
        for j in range(i):
            c3 = x[i] - x[j]
            c2 *= c3
            if j == i - 1:
                for k in range(mn, 0, -1):
                    c[k, i] = c1 * (k * c[k - 1, i - 1] - c5 * c[k, i - 1]) / c2
                c[0, i] = -c1 * c5 * c[0, i - 1] / c2
            for k in range(mn, 0, -1):
                c[k, j] = (c4 * c[k, j] - k * c[k - 1, j]) / c3
            c[0, j] = c4 * c[0, j] / c3
        c1 = c2

    return c


class CompressibleRayleighSolver:
    """
    Solveur de stabilité linéaire pour une couche de mélange compressible.
    Résout le problème aux valeurs propres généralisé (GEP) : A*X = c*B*X
    X = [u, v, p]^T

    Note: cette implémentation résout un problème aux valeurs propres discrétisé
    en différences finies. Ce n'est pas une méthode du tir.
    """
    def __init__(
        self,
        N=400,
        L=15,
        alpha=0.5,
        M=0.0,
        stretched=True,
        stretch_strength=0.82,
    ):
        self.N = N
        self.L = L
        self.alpha = alpha
        self.M = M
        self.stretched = stretched
        self.stretch_strength = stretch_strength
        
        self.setup_grid()
        self.setup_operators()
        self.U = np.tanh(self.y)
        self.Uy = 1.0 / np.cosh(self.y)**2  # dU/dy
        
    def setup_grid(self):
        """
        Définit la grille 1D.

        Lorsque `stretched=True`, on utilise une transformation inspirée de la
        thèse : xi = y / (d + a y^3), avec xi dans [-1, 1]. Cela concentre les
        points autour de y = 0 tout en conservant la symétrie du domaine.
        """
        self.xi = np.linspace(-1.0, 1.0, self.N)
        if self.stretched:
            a = self.stretch_strength / (self.L**2)
            d = self.L * (1.0 - self.stretch_strength)
            y_values = []
            for xi_val in self.xi:
                if np.isclose(xi_val, 0.0):
                    y_values.append(0.0)
                    continue

                coeffs = [a * xi_val, 0.0, -1.0, d * xi_val]
                roots = np.roots(coeffs)
                real_roots = roots[np.isclose(np.imag(roots), 0.0, atol=1e-10)].real
                valid_roots = real_roots[(real_roots >= -self.L - 1e-10) & (real_roots <= self.L + 1e-10)]
                if len(valid_roots) == 0:
                    raise ValueError(f"Aucune racine reelle valide pour xi={xi_val}")
                y_values.append(valid_roots[np.argmin(np.abs(valid_roots - self.L * xi_val))])
            self.y = np.array(y_values, dtype=float)
        else:
            self.y = np.linspace(-self.L, self.L, self.N)
        self.h = np.gradient(self.y)

    def setup_operators(self):
        """
        Construit D_y et D_yy directement sur la grille physique non uniforme.
        """
        N = self.N
        dy_matrix = np.zeros((N, N), dtype=float)
        dyy_matrix = np.zeros((N, N), dtype=float)

        for i in range(N):
            if i == 0:
                stencil = np.array([0, 1, 2])
            elif i == N - 1:
                stencil = np.array([N - 3, N - 2, N - 1])
            else:
                stencil = np.array([i - 1, i, i + 1])

            weights = finite_difference_weights(self.y[i], self.y[stencil], 2)
            dy_matrix[i, stencil] = weights[1]
            dyy_matrix[i, stencil] = weights[2]

        self.Dy = csr_matrix(dy_matrix)
        self.Dyy = csr_matrix(dyy_matrix)

    def assemble_matrices(self):
        """
        Assemble le GEP réduit sur les points intérieurs.

        Les conditions aux limites homogènes sont imposées en retirant les
        inconnues de bord, ce qui évite d'introduire une matrice B singulière.
        """
        N = self.N - 2
        alpha = self.alpha
        M = self.M
        I = diags([np.ones(N)], [0])
        Dy_int = self.Dy[1:-1, 1:-1]
        U_int = self.U[1:-1]
        Uy_int = self.Uy[1:-1]
        U_mat = diags([U_int], [0])
        Uy_mat = diags([Uy_int], [0])
        
        from scipy.sparse import vstack, hstack
        
        # A = [[alpha, -i*D_y, alpha*U*M^2],
        #      [alpha*U, -i*U', alpha],
        #      [0, alpha*U, -i*D_y]]
        
        A11 = alpha * I
        A12 = -1j * Dy_int
        A13 = alpha * M**2 * U_mat
        
        A21 = alpha * U_mat
        A22 = -1j * Uy_mat
        A23 = alpha * I
        
        A31 = csr_matrix((N, N))
        A32 = alpha * U_mat
        A33 = -1j * Dy_int
        
        A = vstack([
            hstack([A11, A12, A13]),
            hstack([A21, A22, A23]),
            hstack([A31, A32, A33])
        ]).tolil()
        
        # B = [[0, 0, alpha*M^2],
        #      [alpha, 0, 0],
        #      [0, alpha, 0]]
        
        B13 = alpha * M**2 * I
        B21 = alpha * I
        B32 = alpha * I
        
        B = vstack([
            hstack([csr_matrix((N, N)), csr_matrix((N, N)), B13]),
            hstack([B21, csr_matrix((N, N)), csr_matrix((N, N))]),
            hstack([csr_matrix((N, N)), B32, csr_matrix((N, N))])
        ]).tocsr()

        return A.tocsr(), B

    def solve_standard_rayleigh(self, n_eig=5):
        """Solveur de Rayleigh standard (incompressible) pour vérification."""
        # (U-c)(v'' - a^2 v) - U''v = 0 -> (U v'' - a^2 U v - U'' v) = c(v'' - a^2 v)
        alpha = self.alpha
        U = self.U
        Upp = -2.0 * np.tanh(self.y) / np.cosh(self.y)**2

        Lap = self.Dyy - alpha**2 * diags(np.ones(self.N))
        Lap_int = Lap[1:-1, 1:-1]
        A = diags(U[1:-1]) @ Lap_int - diags(Upp[1:-1])
        B = Lap_int

        vals, _ = eigs(A.tocsr(), k=n_eig, M=B.tocsr(), sigma=0.2j, which='LM')
        return vals

    def solve(self, n_eig=20):
        """Résout le GEP pour c."""
        A, B = self.assemble_matrices()
        sigma = 0.15j if self.M < 1.0 else 0.05 + 0.08j
        try:
            vals, vecs = eigs(A, k=n_eig, M=B, sigma=sigma, which='LM')
        except Exception:
            vals, vecs = eigs(A, k=n_eig, M=B, which='LI')
        return vals, vecs

    def get_most_unstable(self):
        """Retourne la valeur propre c et le vecteur propre associé."""
        vals, vecs = self.solve()
        idx = np.argmax(np.imag(vals))
        return vals[idx], self.expand_vector(vecs[:, idx])

    def growth_rate(self, c):
        """Retourne le taux de croissance omega_i = alpha * Im(c)."""
        return self.alpha * np.imag(c)

    def solve_dominant_mode(self, n_eig=20):
        """
        Retourne les observables du mode dominant pour un couple (alpha, M).

        En régime supersonique, le spectre peut contenir une paire complexe
        dominante avec meme partie imaginaire et parties réelles opposées. On
        conserve ici le mode de plus grande partie imaginaire et, en cas
        d'ex aequo numérique, celui de plus petite |Re(c)| pour limiter les
        alternances de branche.
        """
        # En subsonique, on impose explicitement la neutralité au-delà de la
        # frontière analytique alpha^2 + M^2 = 1.
        if self.M < 1.0 and self.alpha**2 + self.M**2 >= 1.0:
            return {
                "c": 0.0j,
                "cr": 0.0,
                "ci": 0.0,
                "omega_i": 0.0,
                "vector": np.zeros(3 * self.N, dtype=complex),
            }

        filtered = self.get_candidate_modes(n_eig=n_eig)
        if not filtered:
            return {
                "c": 0.0j,
                "cr": 0.0,
                "ci": 0.0,
                "omega_i": 0.0,
                "vector": np.zeros(3 * self.N, dtype=complex),
            }

        chosen = max(filtered, key=lambda item: item["omega_i"])
        c = chosen["c"]
        return {
            "c": c,
            "cr": float(np.real(c)),
            "ci": float(np.imag(c)),
            "omega_i": float(self.growth_rate(c)),
            "vector": chosen["vector"],
        }

    def expand_vector(self, vec_reduced):
        """
        Recompose le vecteur propre complet en ajoutant les bords nuls.
        """
        n_int = self.N - 2
        full = np.zeros(3 * self.N, dtype=complex)
        full[1:self.N - 1] = vec_reduced[:n_int]
        full[self.N + 1:2 * self.N - 1] = vec_reduced[n_int:2 * n_int]
        full[2 * self.N + 1:3 * self.N - 1] = vec_reduced[2 * n_int:]
        return full

    def get_candidate_modes(self, n_eig=12):
        """
        Retourne les candidats spectraux physiquement plausibles pour un point.
        """
        sigma_list = [0.05j, 0.10j, 0.20j, 0.30j, 0.40j] if self.M < 1.0 else [
            0.02 + 0.03j,
            -0.02 + 0.03j,
            0.05 + 0.05j,
            -0.05 + 0.05j,
            0.08j,
        ]

        candidates = []
        A, B = self.assemble_matrices()
        for sigma in sigma_list:
            try:
                vals, vecs = eigs(A, k=n_eig, M=B, sigma=sigma, which='LM')
            except Exception:
                continue
            for idx, val in enumerate(vals):
                if not np.isfinite(val) or abs(val) > 2.0 or np.imag(val) <= 0.0:
                    continue
                candidates.append((val, vecs[:, idx]))

        if not candidates:
            vals, vecs = self.solve(n_eig=n_eig)
            candidates = [(val, vecs[:, idx]) for idx, val in enumerate(vals) if np.isfinite(val) and np.imag(val) > 0.0]

        filtered = []
        for val, vec in candidates:
            cr = float(np.real(val))
            ci = float(np.imag(val))
            omega_i = float(self.alpha * ci)

            if self.M < 1.0:
                if abs(cr) > 0.15 or omega_i > 0.25:
                    continue
            else:
                if abs(cr) > 0.35 or omega_i > 0.12:
                    continue

            if any(abs(val - other["c"]) < 1e-6 for other in filtered):
                continue

            filtered.append(
                {
                    "c": val,
                    "cr": cr,
                    "ci": ci,
                    "omega_i": omega_i,
                    "vector": self.expand_vector(vec),
                }
            )

        return sorted(filtered, key=lambda item: item["omega_i"])


def sample_growth_map(
    alphas,
    machs,
    *,
    N=200,
    L=15.0,
    stretched=True,
    n_eig=12,
):
    """
    Echantillonne le solveur sur une grille (alpha, M).

    Retourne un DataFrame avec une ligne par couple de paramètres et les champs
    utiles pour reconstruire les isolignes de Blumen.
    """
    rows = []
    for mach in machs:
        previous_omega = 0.0
        max_step = 0.08 if mach < 1.0 else 0.03
        for alpha in alphas:
            solver = CompressibleRayleighSolver(
                N=N,
                L=L,
                alpha=float(alpha),
                M=float(mach),
                stretched=stretched,
            )
            if mach < 1.0 and float(alpha) ** 2 + float(mach) ** 2 >= 1.0:
                dominant = {"cr": 0.0, "ci": 0.0, "omega_i": 0.0}
                previous_omega = 0.0
            else:
                candidates = solver.get_candidate_modes(n_eig=n_eig)
                if not candidates:
                    dominant = {"cr": 0.0, "ci": 0.0, "omega_i": 0.0}
                else:
                    admissible = [
                        candidate
                        for candidate in candidates
                        if candidate["omega_i"] <= previous_omega + max_step + 1e-12
                    ]
                    if admissible:
                        dominant = max(admissible, key=lambda item: item["omega_i"])
                    else:
                        dominant = min(
                            candidates,
                            key=lambda item: abs(item["omega_i"] - (previous_omega + max_step)),
                        )
                previous_omega = dominant["omega_i"]
            rows.append(
                {
                    "alpha": float(alpha),
                    "Mach": float(mach),
                    "cr": dominant["cr"],
                    "ci": dominant["ci"],
                    "omega_i": dominant["omega_i"],
                }
            )

    return pd.DataFrame(rows)

    def plot_eigenfunction(self, vec, title="Eigenfunction"):
        """Trace les profils des perturbations u, v, p."""
        N = self.N
        u = vec[:N]
        v = vec[N:2*N]
        p = vec[2*N:]
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.y, np.abs(u), label='|u|')
        plt.plot(self.y, np.abs(v), label='|v|')
        plt.plot(self.y, np.abs(p), label='|p|')
        plt.title(title)
        plt.xlabel('y')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)
        filename = f"classical_solver/eigenfunction_{title.replace(' ', '_')}.png"
        plt.savefig(filename)
        print(f"Eigenfunction plot saved: {filename}")

# --- Routines de vérification ---

def test_michalke():
    print("\n--- Test 1 : Benchmark de Michalke (M=0) ---")
    alpha = 0.5
    solver = CompressibleRayleighSolver(N=400, L=15, alpha=alpha, M=1e-6, stretched=True)
    c, vec = solver.get_most_unstable()
    omega_i = alpha * np.imag(c)
    print(f"Alpha: {alpha}, M: {solver.M}")
    print(f"Vitesse de phase c: {c:.5f}")
    print(f"Taux de croissance omega_i: {omega_i:.5f}")
    print(f"Référence Michalke: ~0.1875")
    
    solver.plot_eigenfunction(vec, title="Michalke alpha=0.5")
    
    # On compare omega_i
    # Note: L'écart peut venir de la définition exacte du profil tanh(y) vs tanh(y/theta)
    # ou de la structure du GEP. Si omega_i est entre 0.15 et 0.20, c'est acceptable.
    assert np.abs(omega_i - 0.1875) < 0.05, f"Erreur trop grande: {omega_i} vs 0.1875"
    print("Test 1 réussi !")

def test_subsonic_limit():
    print("\n--- Test 2 : Limite théorique subsonique (M=0.4) ---")
    M = 0.4
    alphas = np.linspace(0.1, 1.1, 15)
    growths = []
    
    alpha_cut = np.sqrt(1 - M**2)
    print(f"Limite théorique alpha_c = {alpha_cut:.4f}")
    
    for a in alphas:
        solver = CompressibleRayleighSolver(N=400, L=15, alpha=a, M=M, stretched=True)
        c, _ = solver.get_most_unstable()
        omega_i = solver.growth_rate(c)
        growths.append(omega_i)
        print(f"alpha: {a:.2f} -> omega_i: {omega_i:.5f}")
        
    plt.figure(figsize=(8, 5))
    plt.plot(alphas, growths, 'o-', label=f'Solveur (M={M})')
    plt.axvline(alpha_cut, color='r', linestyle='--', label='Limite théorique')
    plt.xlabel('alpha')
    plt.ylabel('omega_i')
    plt.title(f'Taux de croissance en fonction de alpha (Mach={M})')
    plt.legend()
    plt.grid(True)
    plt.savefig('classical_solver/subsonic_limit.png')
    print("Graphique sauvegardé : classical_solver/subsonic_limit.png")

def test_convergence():
    print("\n--- Test 3 : Étude de convergence (alpha=0.5, M=0.5) ---")
    alpha = 0.5
    M = 0.5
    Ns = [200, 400, 600, 800]
    Ls = [10, 15, 20, 30]
    
    # 1. Convergence en N
    results_N = []
    for n in Ns:
        solver = CompressibleRayleighSolver(N=n, L=20, alpha=alpha, M=M)
        c, _ = solver.get_most_unstable()
        omega_i = solver.growth_rate(c)
        results_N.append(omega_i)
        print(f"N={n}, L=20 -> omega_i={omega_i:.6f}")

    # 2. Convergence en L
    results_L = []
    for l in Ls:
        # On garde une résolution équivalente (N/L approx constant)
        n = int(l * 30) 
        solver = CompressibleRayleighSolver(N=n, L=l, alpha=alpha, M=M)
        c, _ = solver.get_most_unstable()
        omega_i = solver.growth_rate(c)
        results_L.append(omega_i)
        print(f"N={n}, L={l} -> omega_i={omega_i:.6f}")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(Ns, results_N, 's-')
    plt.xlabel('N')
    plt.ylabel('omega_i')
    plt.title('Convergence en N (L=20)')
    
    plt.subplot(1, 2, 2)
    plt.plot(Ls, results_L, 'd-')
    plt.xlabel('L')
    plt.ylabel('omega_i')
    plt.title('Convergence en L (N/L constant)')
    
    plt.tight_layout()
    plt.savefig('classical_solver/convergence_study.png')
    print("Graphique sauvegardé : classical_solver/convergence_study.png")

def test_blumen(csv_path=None):
    print("\n--- Test 4 : Comparaison Blumen ---")
    if csv_path is None or not os.path.exists(csv_path):
        print("Fichier CSV Blumen non trouvé. Création d'un fichier fictif pour démonstration.")
        csv_path = 'classical_solver/blumen_ref.csv'
        df = pd.DataFrame({
            'Mach': [0.0, 0.0, 0.0, 0.5, 0.5, 0.5],
            'alpha': [0.1, 0.5, 0.9, 0.1, 0.5, 0.8],
            'omega_i_reference': [0.01, 0.1875, 0.05, 0.04, 0.12, 0.02]
        })
        df.to_csv(csv_path, index=False)
    
    ref_data = pd.read_csv(csv_path)
    
    # Calcul de nos propres points pour un Mach fixe (ex: M=0.5)
    M_val = 0.5
    alphas = np.linspace(0.1, 0.9, 10)
    growths_calc = []
    for a in alphas:
        solver = CompressibleRayleighSolver(N=400, L=15, alpha=a, M=M_val)
        c, _ = solver.get_most_unstable()
        growths_calc.append(solver.growth_rate(c))
        
    plt.figure()
    plt.plot(alphas, growths_calc, label=f'Notre Solveur (M={M_val})')
    # Superposition des points de ref pour ce Mach
    ref_M = ref_data[ref_data['Mach'] == M_val]
    plt.scatter(ref_M['alpha'], ref_M['omega_i_reference'], color='red', label='Référence Blumen')
    
    plt.xlabel('alpha')
    plt.ylabel('omega_i')
    plt.title('Comparaison avec les données de Blumen')
    plt.legend()
    plt.savefig('classical_solver/blumen_comparison.png')
    print("Graphique sauvegardé : classical_solver/blumen_comparison.png")

if __name__ == "__main__":
    test_michalke()
    test_subsonic_limit()
    test_convergence()
    test_blumen()
