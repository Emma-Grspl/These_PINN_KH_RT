# Protocole PINN subsonique M=0.5, alpha in [0.1, 0.8], physique pure

## Objectif

Construire la baseline **physique pure** qui servira ensuite de reference de comparaison contre les futurs PINNs hybrides a supervision `c_i` clairsemee.

Le but de ce run n'est pas de "gagner" a tout prix sur `c_i`, mais de mesurer jusqu'ou vont :

- la PDE ;
- les conditions aux bords Riccati ;
- les contraintes modales locales ;
- et une regularisation spectrale minimale non supervisee.

## Domaine du run

- `Mach = 0.5`
- `alpha in [0.1, 0.8]`
- formulation modale : `riccati`
- aucune supervision classique dans la loss sur `c_i`

## Configuration retenue

### Architecture

- `hidden_dim = 160`
- `mode_depth = 4`
- `ci_depth = 2`
- `mode_experts = 2`
- `alpha_split_threshold = 0.40`
- `mapping_scale = 3.0`

Cette architecture est choisie pour rester reutilisable plus tard dans la comparaison a architecture egale contre les runs hybrides `4 / 8 / 16` points.

### Echantillonnage

- `n_interior = 512`
- `n_boundary = 64`
- `n_alpha_supervision = 128`
- `n_anchor_alpha = 32`
- `n_norm_interior = 256`
- `n_reference_alpha = 121`
- `n_audit_alpha = 31`
- `n_mode_audit_alpha = 11`
- `n_mode_audit_y = 801`

### Loss active

- `w_pde = 1.0`
- `w_bc_kappa = 10.0`
- `w_bc_q = 20.0`
- `w_riccati_center_kappa = 5.0`
- `w_riccati_center_peak = 2.0`
- `w_riccati_boundary_band_kappa = 2.0`
- `w_riccati_boundary_band_q = 8.0`
- `w_ci_low_alpha_zero = 10.0`
- `w_ci_smoothness = 0.5`

### Loss desactivee

- `w_ci_supervision = 0.0`
- `w_q_supervision = 0.0`
- `w_riccati_anchor = 0.0`
- `w_ci_stability_outside = 0.0`
- `w_ci_neutrality = 0.0`
- `w_riccati_shooting_match = 0.0`
- `w_riccati_shooting_path = 0.0`
- `w_riccati_ci_local_min = 0.0`

Le choix est deliberement conservateur :

- on garde une baseline purement PINN ;
- on evite les termes de shooting encore trop couteux pour un sweep large en `alpha` ;
- on n'impose pas de contrainte de neutralite hors bande puisque `alpha_max = 0.8 < alpha_n(M=0.5)`.

## Protocole d'execution

### Job nominal

- `epochs = 5000`
- `learning_rate = 1e-3`
- `audit_every = 100`
- `checkpoint_every = 500`
- budget slurm : `20h`

### Critere pratique d'arret

Le trainer actuel n'a pas d'early stopping automatique. Le critere retenu est donc **manuel** a partir de `history.csv`.

On considere que le run a stagne si, sur les `5` derniers audits :

- `audit_ci_mae` n'ameliore plus de facon visible ;
- `audit_checkpoint_metric` plafonne ;
- les metriques modales (`audit_p_rel_l2_mean`, `audit_env_rel_mean`) ne s'ameliorent plus non plus.

Dans ce cas :

- on garde `model_best.pt` comme baseline pure physique ;
- on n'insiste pas davantage sur cette branche ;
- on passe ensuite au protocole hybride sparse.

## Artefacts attendus

Dans le dossier de sortie :

- `history.csv`
- `config.csv`
- `model_best.pt`
- `checkpoint_epoch_*.pt`

## Interpretation attendue

Le resultat plausible de cette baseline est :

- des modes partiellement plausibles ;
- une courbe `c_i(alpha)` plus lisse qu'un pur hasard ;
- mais encore un ecart notable a la reference classique.

C'est precisement cette baseline qui servira a quantifier le gain apporte plus tard par :

- `4` points de supervision `c_i`
- puis `8`
- puis `16`

## Fichiers de lancement

- script runner : [run_kh_subsonic_M05_alpha010_080_pure_physics_reference.sh](/Users/emma.grospellier/Thèse/These_PINN_KH_RT/scripts/run_kh_subsonic_M05_alpha010_080_pure_physics_reference.sh)
- slurm Jean Zay : [jz_submit_kh_subsonic_pinn_M05_alpha010_080_pure_physics_reference.slurm](/Users/emma.grospellier/Thèse/These_PINN_KH_RT/launch/jz_submit_kh_subsonic_pinn_M05_alpha010_080_pure_physics_reference.slurm)
