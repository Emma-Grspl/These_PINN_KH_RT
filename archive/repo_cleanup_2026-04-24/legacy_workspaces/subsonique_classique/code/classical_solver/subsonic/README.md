# Workflow Subsonique De Référence

Ce dossier contient plusieurs solveurs de tir subsoniques. La version de référence
à utiliser pour la thèse est désormais :

- [hybrid_subsonic_scan.py](/Users/emma.grospellier/Thèse/These_PINN_KH_RT/classical_solver/subsonic/hybrid_subsonic_scan.py)

## Principe

Le workflow subsonique verrouillé combine deux solveurs :

- [shooting_subsonic.py](/Users/emma.grospellier/Thèse/These_PINN_KH_RT/classical_solver/subsonic/shooting_subsonic.py)
  Solveur principal, rapide, utilisé sur la majorité des points.
- [mstab17_subsonic_solver.py](/Users/emma.grospellier/Thèse/These_PINN_KH_RT/classical_solver/subsonic/mstab17_subsonic_solver.py)
  Solveur de contrôle plus robuste près de la neutralité.

Le script hybride :
- résout toute la grille avec le solveur principal ;
- corrige seulement une bande proche de la frontière neutre ;
- produit la carte finale de référence.

## Statut

Ce workflow est considéré comme la référence subsonique actuelle.

Sorties de référence :
- [subsonic_hybrid_growth_map.csv](/Users/emma.grospellier/Thèse/These_PINN_KH_RT/assets/blumen_shooting_hybrid/subsonic_hybrid_growth_map.csv)
- [subsonic_hybrid_vs_blumen.png](/Users/emma.grospellier/Thèse/These_PINN_KH_RT/assets/blumen_shooting_hybrid/subsonic_hybrid_vs_blumen.png)
- [subsonic_hybrid_error_summary.json](/Users/emma.grospellier/Thèse/These_PINN_KH_RT/assets/blumen_shooting_hybrid/subsonic_hybrid_error_summary.json)

Le dernier run validé donne :
- `global_mae_omega ≈ 1.75e-3`
- `global_median_distance ≈ 4.61e-3`

## Scripts

Scripts principaux :
- [hybrid_subsonic_scan.py](/Users/emma.grospellier/Thèse/These_PINN_KH_RT/classical_solver/subsonic/hybrid_subsonic_scan.py)
  Scan de référence.
- [robust_subsonic_shooting.py](/Users/emma.grospellier/Thèse/These_PINN_KH_RT/classical_solver/subsonic/robust_subsonic_shooting.py)
  Solveur point-à-point combinant les deux méthodes.

Scripts de support :
- [compare_subsonic_shooting_solvers.py](/Users/emma.grospellier/Thèse/These_PINN_KH_RT/classical_solver/subsonic/compare_subsonic_shooting_solvers.py)
  Comparaison entre les deux solveurs.
- [plot_subsonic_error_map.py](/Users/emma.grospellier/Thèse/These_PINN_KH_RT/classical_solver/subsonic/plot_subsonic_error_map.py)
  Carte d’erreur.
- [plot_subsonic_ci_map.py](/Users/emma.grospellier/Thèse/These_PINN_KH_RT/classical_solver/subsonic/plot_subsonic_ci_map.py)
  Visualisation de `c_i`.

Scripts historiques conservés :
- [reconstruct_blumen_subsonic_shooting.py](/Users/emma.grospellier/Thèse/These_PINN_KH_RT/classical_solver/subsonic/reconstruct_blumen_subsonic_shooting.py)
- [reconstruct_blumen_subsonic_robust.py](/Users/emma.grospellier/Thèse/These_PINN_KH_RT/classical_solver/subsonic/reconstruct_blumen_subsonic_robust.py)

Ils restent utiles pour comparaison, mais ne sont plus le point d’entrée recommandé.

## Lancement Local

Petit test :

```bash
python3 classical_solver/subsonic/hybrid_subsonic_scan.py --num-mach 9 --num-alpha 9
```

Carte plus fine :

```bash
python3 classical_solver/subsonic/hybrid_subsonic_scan.py --num-mach 41 --num-alpha 41
```

## Lancement Jean Zay

Script Slurm de référence :
- [jz_submit_subsonic.slurm](/Users/emma.grospellier/Thèse/These_PINN_KH_RT/launch/jz_submit_subsonic.slurm)

Commande :

```bash
sbatch launch/jz_submit_subsonic.slurm
```

## Convention de travail

Pour la suite de la thèse :
- on considère le subsonique comme verrouillé avec ce workflow hybride ;
- les nouveaux développements portent d’abord sur le supersonique ;
- si une modification affecte le subsonique, elle doit être comparée à cette référence.
