# Protocole PINN subsonique `alpha-Mach`

Ce document fixe le premier pilote 2D subsonique en `(alpha, Mach)`.

## Objectif

- passer d'un PINN valide sur des lignes `Mach` fixes vers un PINN qui depend explicitement de `(alpha, Mach)`
- conserver une reconstruction spectrale credibe sur une bande Mach etroite
- verifier que la reconstruction modale reste exploitable sur quelques coupes representatives de cette bande

L'objectif final reste bien un balayage `alpha-Mach`, pas une collection de runs 1D independants. Les sections 1D servent ici de bootstrap.

## Etat de depart

Les sections 1D de bootstrap sont maintenant suffisantes pour ouvrir un premier pilote 2D :

- reference figee `M=0.5` :
  - [frozen_M05_riccati_reference_current](/Users/emma.grospellier/Thèse/These_PINN_KH_RT/assets/pinn_subsonic/mach_fixed/frozen_M05_riccati_reference_current)
- section validee `M=0.6` :
  - [experiment_M06_mode_repair_edges_2026-06-05](/Users/emma.grospellier/Thèse/These_PINN_KH_RT/assets/pinn_subsonic/experiment_M06_mode_repair_edges_2026-06-05)

Ces deux sections ne remplacent pas le 2D. Elles fournissent :

- une architecture stable
- des poids de bootstrap
- deux coupes Mach de reference pour lire la coherence du premier band 2D

## Bande 2D pilote

Le premier domaine retenu est volontairement etroit :

- `alpha in [0.05, 0.75]`
- `Mach in [0.50, 0.60]`

Ce choix isole la difficulte "dependance en Mach" sans ajouter trop d'extrapolation. Il permet aussi d'utiliser `M=0.5` comme warm start naturel et `M=0.6` comme seconde coupe de validation.

## Question scientifique

Le point a trancher est le suivant :

- un PINN initialise sur une section 1D `Mach = 0.5` peut-il apprendre une surface `c_i(alpha, Mach)` credible sur `[0.5, 0.6]` tout en gardant des modes exploitables sur quelques points intermediaires, en particulier vers `Mach = 0.55` ?

## Role du classique

Pour ce premier pilote 2D, le classique reste la reference de comparaison sur `c_i`. C'est un choix de bootstrap, pas l'etat final vise.

On utilise le classique pour :

- construire la surface de reference `c_i(alpha, Mach)`
- mesurer les erreurs spectrales 2D
- auditer les modes sur un petit ensemble de couples `(alpha, Mach)`

On n'utilise pas le classique pour imposer une supervision modale exhaustive sur toute la surface.

## Warm start retenu

Le pilote 2D demarre depuis la reference 1D `M=0.5`.

Concretement :

- on reprend l'architecture du run 1D fige
- on initialise le modele 2D avec les poids du modele 1D
- les poids associes a l'entree `Mach` sont initialises a zero au premier layer

Donc, au debut de l'entrainement, le modele 2D se comporte comme le modele 1D `M=0.5`, puis apprend progressivement la variation en `Mach`.

## Configuration pilote recommandee

### Entrainement

- runner :
  - [run_kh_subsonic_pinn_2d_pilot_M05_M06.py](/Users/emma.grospellier/Thèse/These_PINN_KH_RT/scripts/run_kh_subsonic_pinn_2d_pilot_M05_M06.py)
- launcher Jean Zay :
  - [jz_submit_kh_subsonic_pinn_2d_pilot_M05_M06.slurm](/Users/emma.grospellier/Thèse/These_PINN_KH_RT/launch/jz_submit_kh_subsonic_pinn_2d_pilot_M05_M06.slurm)

Defauts du pilote :

- `epochs = 2500`
- `n_interior = 384`
- `n_boundary = 64`
- `n_supervision = 96`
- `n_reference_alpha = 31`
- `n_reference_mach = 7`
- separate branch optimizers actives
- representation modale `riccati`
- supervision classique dense sur `c_i` maintenue pour ce premier band

### Evaluation spectrale

Sorties attendues :

- `subsonic_pinn_alphamach_ci_surface.csv`
- `subsonic_pinn_alphamach_ci_error_heatmap.png`
- `subsonic_pinn_alphamach_ci_isolines_overlay.png`
- `subsonic_pinn_alphamach_ci_isolines_with_error.png`
- `subsonic_pinn_alphamach_training_summary.csv`

### Audit modal

Le premier audit modal reste localise. Les points par defaut sont :

- `(0.10, 0.50)`
- `(0.25, 0.50)`
- `(0.65, 0.50)`
- `(0.10, 0.55)`
- `(0.25, 0.55)`
- `(0.65, 0.55)`
- `(0.10, 0.60)`
- `(0.25, 0.60)`
- `(0.65, 0.60)`

Sorties attendues :

- `subsonic_pinn_alphamach_mode_error_heatmaps.csv`
- `subsonic_pinn_alphamach_mode_error_heatmaps.png`
- `subsonic_pinn_alphamach_mode_points.csv`
- `subsonic_pinn_alphamach_modes.pdf`

## Ordre de lecture

La lecture du pilote 2D se fait dans cet ordre :

1. surface `c_i` PINN vs classique
2. heatmap d'erreur `c_i`
3. isolignes superposees
4. heatmaps modales locales
5. planches de modes sur les 9 points d'audit

Il ne faut pas inverser cet ordre. Le pilote est d'abord juge sur la coherence spectrale 2D.

## Critere de reussite

Le pilote sera considere comme suffisamment bon si :

- les isolignes `c_i` restent coherentes sur tout `[0.5, 0.6]`
- l'erreur sur `c_i` reste concentree et lisible
- les points modaux a `Mach = 0.55` restent exploitables
- les coupes `Mach = 0.5` et `Mach = 0.6` restent compatibles avec les sections 1D deja validees

## Ce qu'on ne demande pas encore

Pour ce premier pilote 2D, on ne demande pas encore :

- une perfection uniforme sur toute la surface modale
- une reduction drastique de l'usage du classique
- un elargissement immediat a `Mach = 0.7`
- un passage au supersonique

La priorite est d'obtenir un premier band `alpha-Mach` propre, relancable et lisible.

## Suite logique si le pilote est bon

Si le pilote `[0.5, 0.6]` est bon, la suite naturelle est :

- figer ce run 2D comme premiere reference `alpha-Mach`
- elargir la bande Mach
- puis seulement tester des variantes avec moins de supervision classique sur `c_i`

Autrement dit : on valide d'abord la generalisation 2D, puis on reduit la dependance au solveur classique.
