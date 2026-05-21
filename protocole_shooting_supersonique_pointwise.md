# Protocole shooting supersonique pointwise

## Objectif

Avant tout balayage global, valider le shooting supersonique sur une liste finie de couples `(alpha, Mach)`.

Le but est de separer deux questions :

1. le solveur local retrouve-t-il une branche propre pour un point donne ;
2. les echecs du paquet global viennent-ils du solveur lui-meme ou du suivi de branche / de l'amorcage.

## Strategie

- lancer un audit par points independants, pas une grille globale ;
- traiter chaque point avec un multistart local ;
- executer les points en parallele CPU ;
- sortir des metriques de succes explicites ;
- conserver les champs du meilleur candidat pour inspection visuelle.

## Metriques a regarder

Pour chaque point :

- `best_spectral_success`
- `best_mode_success`
- `best_success`
- `best_stage1_mismatch`
- `best_stage2_mismatch`
- `best_shooting_cr`
- `best_shooting_ci`
- `best_ln_p_start_right`
- `best_y_limit`
- `left_boundary_amp_fraction`
- `right_boundary_amp_fraction`
- `edge_amp_fraction_max`
- `center8_mass_fraction`
- `left_mass_fraction`
- `right_mass_fraction`
- `peak_y`

Si Blumen est disponible localement :

- `best_err_cr_abs`
- `best_err_ci_abs`
- `best_err_ci_rel`

## Lecture des statuts

- `validated`
  - `spectral_success=True` et `mode_success=True`
- `spectral_only`
  - l'eigenvaleur est acceptable, mais le mode ne raccorde pas correctement
- `mode_only`
  - cas atypique ; a surveiller comme anomalie
- `failed`
  - ni la partie spectrale ni la partie modale ne sont suffisamment propres
- `exception`
  - le solveur ou la reconstruction a plante pour ce point

## Critere pratique

Un point est considere comme reussi si :

- `best_success=True`
- `best_stage1_mismatch < 5e-2`
- `best_stage2_mismatch < 1e-2`
- et les amplitudes aux bords restent faibles

## Fichiers produits

Le batch pointwise produit :

- `<stem>_summary.csv`
- `<stem>_candidates.csv`
- `<stem>_fields.csv`
- `<stem>_status_map.png`
- `<stem>_diagnostics.png`
- `<stem>_modes.pdf`

## Usage

Commande Jean Zay :

```bash
sbatch launch/jz_submit_supersonic_shooting_point_batch.slurm
```

Exemple de liste de points :

```bash
POINTS="0.18:1.33 0.20:1.20 0.20:1.30 0.10:1.25 0.30:1.40 0.50:1.15"
```

## Decision ensuite

- si une famille de points critiques est propre, on peut envisager une continuation locale ;
- si beaucoup de points `validated` existent deja, alors seulement on revient vers un balayage plus global ;
- sinon, on revoit l'amorcage, l'ordre de continuation et/ou les boites de recherche.
