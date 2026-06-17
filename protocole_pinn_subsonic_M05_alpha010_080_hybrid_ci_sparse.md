# Protocole PINN subsonique M=0.5, alpha in [0.1, 0.8], hybride `c_i` sparse

## Objectif

Comparer a architecture egale la baseline **physique pure** contre trois variantes **hybrides** ou l'on supervise uniquement `c_i` avec un petit nombre d'ancres classiques fixes.

Les trois budgets testes sont :

- `4` points
- `8` points
- `16` points

## Principe de comparaison

La comparaison est volontairement stricte :

- meme architecture ;
- meme optimizer ;
- meme domaine `Mach = 0.5`, `alpha in [0.1, 0.8]` ;
- memes termes physiques modaux ;
- meme budget d'epochs ;
- seule difference : la presence d'une supervision classique sparse sur `c_i`.

Important :

- ici, `4 / 8 / 16 points` signifie **4 / 8 / 16 ancres fixes en alpha**, vues a chaque epoch par le terme `loss_ci` ;
- ce n'est **pas** un budget de points aleatoires reechantillonnes a chaque epoch ;
- il s'agit donc bien d'une supervision sparse au sens experimental utile pour l'article.

## Architecture et loss physique

On reprend exactement la baseline pure physique definie dans :

- [protocole_pinn_subsonic_M05_alpha010_080_pure_physics.md](/Users/emma.grospellier/Thèse/These_PINN_KH_RT/protocole_pinn_subsonic_M05_alpha010_080_pure_physics.md)

Les termes physiques conserves sont :

- `w_pde = 1.0`
- `w_bc_kappa = 10.0`
- `w_bc_q = 20.0`
- `w_riccati_center_kappa = 5.0`
- `w_riccati_center_peak = 2.0`
- `w_riccati_boundary_band_kappa = 2.0`
- `w_riccati_boundary_band_q = 8.0`
- `w_ci_low_alpha_zero = 10.0`
- `w_ci_smoothness = 0.5`

Les aides classiques modales restent desactivees :

- `w_riccati_anchor = 0.0`
- `w_q_supervision = 0.0`
- pas de supervision classique de mode
- pas de `shooting_match`
- pas de `shooting_path`

## Supervision `c_i`

Le terme supplementaire est :

- `w_ci_supervision = 5.0`

et il est applique uniquement sur un ensemble fixe d'ancres en `alpha`.

## Choix des ancres

On utilise une grille maitre de `16` points, puis les budgets `8` et `4` sont des sous-ensembles de cette meme grille. Cela rend la comparaison plus propre.

### Grille 16 points

`0.100, 0.147, 0.193, 0.240, 0.287, 0.333, 0.380, 0.427, 0.473, 0.520, 0.567, 0.613, 0.660, 0.707, 0.753, 0.800`

### Grille 8 points

`0.100, 0.193, 0.287, 0.380, 0.520, 0.613, 0.707, 0.800`

### Grille 4 points

`0.100, 0.287, 0.520, 0.800`

## Pourquoi ce choix

- couverture du bas de bande, du centre et du haut de bande ;
- jeux emboites pour une comparaison propre ;
- nombre limite de points pour rester dans un regime vraiment sparse.

## Protocole d'execution

### Hyperparametres communs

- `epochs = 5000`
- `learning_rate = 1e-3`
- `audit_every = 100`
- `checkpoint_every = 500`
- budget slurm : `20h`

### Critere pratique d'arret

Le trainer n'a pas d'early stopping automatique.

On examine `history.csv` et on arrete manuellement si :

- `audit_ci_mae` plafonne sur plusieurs audits ;
- et que les metriques modales cessent elles aussi de s'ameliorer.

## Ce que l'on compare a la fin

Pour chaque run :

- `audit_ci_mae`
- `audit_ci_max_abs`
- `audit_p_rel_l2_mean`
- `audit_env_rel_mean`
- `audit_phase_rel_mean`
- `audit_checkpoint_metric`

Le resultat attendu est :

- `4 points` : probablement encore limite pour verrouiller toute la branche ;
- `8 points` : candidat plausible pour le minimum utile ;
- `16 points` : verrouillage nettement plus stable.

## Fichiers de lancement

- runner commun : [run_kh_subsonic_M05_alpha010_080_ci_sparse_reference.sh](/Users/emma.grospellier/Thèse/These_PINN_KH_RT/scripts/run_kh_subsonic_M05_alpha010_080_ci_sparse_reference.sh)
- slurm 4 points : [jz_submit_kh_subsonic_pinn_M05_alpha010_080_hybrid_ci4_reference.slurm](/Users/emma.grospellier/Thèse/These_PINN_KH_RT/launch/jz_submit_kh_subsonic_pinn_M05_alpha010_080_hybrid_ci4_reference.slurm)
- slurm 8 points : [jz_submit_kh_subsonic_pinn_M05_alpha010_080_hybrid_ci8_reference.slurm](/Users/emma.grospellier/Thèse/These_PINN_KH_RT/launch/jz_submit_kh_subsonic_pinn_M05_alpha010_080_hybrid_ci8_reference.slurm)
- slurm 16 points : [jz_submit_kh_subsonic_pinn_M05_alpha010_080_hybrid_ci16_reference.slurm](/Users/emma.grospellier/Thèse/These_PINN_KH_RT/launch/jz_submit_kh_subsonic_pinn_M05_alpha010_080_hybrid_ci16_reference.slurm)
