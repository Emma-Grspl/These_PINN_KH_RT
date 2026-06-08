# Protocole PINN subsonique `alpha-Mach`

Ce document fixe la strategie de construction du PINN subsonique global en `(alpha, Mach)`.

## Objectif

- obtenir a terme un PINN qu'on interroge avec `alpha, Mach` pour predire `c_i(alpha, Mach)`
- et qu'on interroge avec `xi, alpha, Mach` pour reconstruire le mode associe
- passer d'un PINN valide sur des lignes `Mach` fixes vers un PINN qui depend explicitement de `(alpha, Mach)`
- conserver une reconstruction spectrale credible sur une bande Mach etroite
- verifier que la reconstruction modale reste exploitable sur quelques coupes representatives de cette bande

L'objectif final reste bien un balayage `alpha-Mach`, pas une collection de runs 1D independants. Les sections 1D servent ici de bootstrap.

La bande finale visee en subsonique reste large :

- `Mach in [0.1, 0.8]`

Le point important est donc le suivant :

- le modele final doit etre global en `(alpha, Mach)`
- mais l'apprentissage n'a pas besoin d'etre "from scratch sur toute la bande"
- il peut et doit etre progressif

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

## Ce qui marche deja

Le pilote 2D `[0.5, 0.6]` est un point d'appui valide :

- la surface `c_i(alpha, Mach)` est propre sur cette bande
- les modes restent exploitables sur les points d'audit
- l'interpolation a `Mach = 0.55` est deja credible

Le band etendu `[0.5, 0.7]` est prometteur mais pas encore fige :

- le spectral reste correct
- une partie de la qualite modale se degrade sur la bande deja acquise
- il faut donc stabiliser l'extension avant de l'utiliser comme nouvelle reference

## Ce qui ne marche pas en l'etat

Le run purement physique fixe-Mach suivant doit etre considere comme un test negatif informatif :

- job `1936761`
- date de lancement : `2026-06-05`
- protocole : `M = 0.5`, `alpha in [0.05, 0.85]`
- loss active :
  - residu PDE Riccati
  - BC asymptotiques
  - contrainte au centre
  - contrainte de bande aux bords
  - matching gauche-droite par shooting
  - critere de minimum local en `c_i`
  - aucune supervision classique dense sur `c_i`

Ce run montre que la formulation actuelle ne permet pas d'apprendre proprement toute la courbe `c_i(alpha)` a `Mach` fixe en pur PINN :

- `ci_mae` reste tres grand : `2.329e-01 -> 1.819e-01` entre `epoch 1` et `epoch 300`
- `p_rel` reste mauvais : environ `0.32 - 0.40`
- `n_focus = 8` tout du long
- le job est coupe au walltime le `2026-06-06` sans entrer dans une zone utile

La conclusion methodologique est nette :

- la physique actuelle peut aider a reconstruire un eigenpair
- mais elle ne selectionne pas encore assez bien une branche complete `c_i(alpha)` from scratch
- le probleme n'est donc pas seulement la generalisation 2D
- il apparait deja sur une simple ligne 1D a `Mach` fixe

Ce point ne contredit pas l'objectif final global en `(alpha, Mach)`. Il contraint seulement la strategie d'apprentissage :

- il ne faut pas demander au PINN global d'apprendre toute la famille spectrale d'un seul coup en pur physique
- il faut construire cette famille progressivement

## Bande 2D pilote

Le premier domaine retenu est volontairement etroit :

- `alpha in [0.05, 0.75]`
- `Mach in [0.50, 0.60]`

Ce choix isole la difficulte "dependance en Mach" sans ajouter trop d'extrapolation. Il permet aussi d'utiliser `M=0.5` comme warm start naturel et `M=0.6` comme seconde coupe de validation.

## Question scientifique

Le point a trancher est le suivant :

- un PINN initialise sur une section 1D `Mach = 0.5` peut-il apprendre une surface `c_i(alpha, Mach)` credible sur `[0.5, 0.6]` tout en gardant des modes exploitables sur quelques points intermediaires, en particulier vers `Mach = 0.55` ?

Cette premiere question a maintenant recu une reponse positive.

La question suivante devient :

- comment passer de ce premier band valide a un PINN global sur `Mach in [0.1, 0.8]`
- sans recommencer a zero
- et sans demander a la physique seule de retrouver directement toute la famille `c_i(alpha, Mach)` from scratch

## Role du classique

Pour ce premier pilote 2D, le classique reste la reference de comparaison sur `c_i`. C'est un choix de bootstrap, pas l'etat final vise.

Dans la version operationnelle actuelle, on utilise le classique pour :

- construire la surface de reference `c_i(alpha, Mach)`
- mesurer les erreurs spectrales 2D
- auditer les modes sur un petit ensemble de couples `(alpha, Mach)`

On n'utilise pas le classique pour imposer une supervision modale exhaustive sur toute la surface.

Le test negatif `1936761` impose une clarification :

- si l'objectif est un PINN global final en `(alpha, Mach)`, le classique peut encore servir au bootstrap
- ce qu'il ne faut plus faire, c'est demander a un `ci_net(alpha)` ou `ci_net(alpha, Mach)` from scratch de remplacer directement toute la famille spectrale sans phase de construction prealable

Le classique a donc deux roles distincts :

- role de reference scientifique
- role de bootstrap operationnel

La reduction de cette dependance doit se faire progressivement, pas en coupant d'un seul coup toute supervision spectrale globale.

## Warm start retenu

Le pilote 2D demarre depuis la reference 1D `M=0.5`.

Concretement :

- on reprend l'architecture du run 1D fige
- on initialise le modele 2D avec les poids du modele 1D
- les poids associes a l'entree `Mach` sont initialises a zero au premier layer

Donc, au debut de l'entrainement, le modele 2D se comporte comme le modele 1D `M=0.5`, puis apprend progressivement la variation en `Mach`.

## Protocole progressif recommande

Le protocole retenu pour converger vers le modele final global est le suivant.

### Niveau 1 - Eigenpairs purs PINN locaux

Objectif :

- verifier qu'un PINN purement physique sait retrouver un eigenpair local
- sans demander d'un coup toute la courbe `c_i(alpha)` ou toute la surface `c_i(alpha, Mach)`

Configuration :

- `alpha` fixe
- `Mach` fixe
- `c_i` scalaire
- matching physique actif
- aucune supervision classique dans la loss

Script de depart :

- [run_kh_subsonic_singlecase_pure_physics_scalar_ci_matching.sh](/Users/emma.grospellier/Thèse/These_PINN_KH_RT/scripts/run_kh_subsonic_singlecase_pure_physics_scalar_ci_matching.sh)

But :

- etablir ou non qu'un eigenpair isole est identifiable en pur PINN
- mesurer la robustesse en fonction de `alpha` et `Mach`

### Niveau 2 - Continuation pure PINN a Mach fixe

Objectif :

- construire une branche `c_i(alpha)` par continuation
- au lieu de la regresser d'un seul coup

Principe :

- on choisit un point de depart `(alpha_0, M)`
- on converge en pur PINN sur ce point
- on warmstart le point voisin `alpha_1`
- on poursuit point par point sur la ligne

Ce niveau est celui qui doit remplacer, pour les tests purs PINN, le run global fixe-Mach de type `1936761`.

Ce qu'on cherche a apprendre ici n'est plus :

- une fonction complete `c_i(alpha)` from scratch

mais :

- une suite d'eigenpairs relies par continuation

### Niveau 3 - Continuation en Mach

Objectif :

- etendre les branches construites en `alpha` vers d'autres `Mach`

Principe :

- partir d'une ligne stable a `Mach = M_k`
- warmstarter la ligne voisine `Mach = M_{k+1}`
- propager de proche en proche

Pour la bande subsonique visee, l'ordre recommande est :

1. consolider `M = 0.5`
2. consolider `M = 0.6`
3. ouvrir `M = 0.4` et `M = 0.7`
4. seulement ensuite etendre vers `M = 0.3`, `0.2`, `0.1` et `0.8`

L'idee est de ne pas demander au reseau global de couvrir d'emblee `Mach in [0.1, 0.8]`.

### Niveau 4 - Distillation dans le PINN global final

Objectif :

- construire le vrai modele final `(\xi, alpha, M) -> mode` et `(alpha, M) -> c_i`

Principe :

- on entraine le reseau global a partir d'une famille d'eigenpairs deja construite
- cette famille peut etre issue :
  - de continuations pures PINN locales
  - de sections hybrides deja validees
  - ou d'un melange des deux

Le point important est :

- le modele final peut etre global en `(alpha, Mach)`
- sans avoir ete appris from scratch sur toute la bande

### Niveau 5 - Allongement progressif de la bande Mach

Le reseau global final ne doit pas etre ouvert directement sur `Mach in [0.1, 0.8]`.

L'ordre recommande est :

1. `[0.5, 0.6]`
2. `[0.5, 0.7]`
3. `[0.4, 0.7]`
4. `[0.3, 0.7]`
5. `[0.2, 0.8]`
6. `[0.1, 0.8]`

Chaque extension doit etre jugee sur deux criteres separes :

- coherence spectrale globale des isolignes `c_i`
- maintien de la qualite modale sur la bande deja acquise

Si une extension degrade la bande precedente, elle n'est pas promue telle quelle.

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

Cette etape est maintenant depassee :

- le pilote `[0.5, 0.6]` est valide
- le probleme ouvert n'est plus "peut-on faire du 2D"
- c'est "comment etendre proprement ce 2D jusqu'a `Mach in [0.1, 0.8]`"

## Suite logique si le pilote est bon

Si le pilote `[0.5, 0.6]` est bon, la suite naturelle est :

- figer ce run 2D comme premiere reference `alpha-Mach`
- utiliser ce band comme base de continuation vers `Mach = 0.7`
- tester en parallele les eigenpairs purs PINN locaux
- remplacer les essais "full curve pure physics from scratch" par des continuations locales
- allonger ensuite progressivement la bande Mach vers `Mach in [0.1, 0.8]`

Autrement dit :

- on garde comme produit final un PINN global interrogeable avec `alpha, Mach`
- mais on le construit par bootstrap, continuation et allongement de bande
- on ne redemande plus a un unique run pur physique fixe-Mach de decouvrir from scratch toute une famille spectrale complete
