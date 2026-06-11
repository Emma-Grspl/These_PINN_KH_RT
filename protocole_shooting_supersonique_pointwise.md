# Protocole shooting supersonique pointwise

## Etat actuel

- un point de reference est deja valide :
  - `alpha = 0.18`
  - `Mach = 1.33`
- ce point montre que le shooting local peut retrouver une branche propre et une forme modale credible quand l'amorcage est bon ;
- en revanche, tous les points testes ne sont pas valides ;
- on n'a donc pas encore une base suffisante pour reconstruire directement des isolignes globales `c_i` et `c_r`.

Conclusion pratique :

- le shooting supersonique local est utilisable ;
- le protocole global de balayage / suivi de branche n'est pas encore suffisamment robuste ;
- la prochaine etape n'est pas de refaire des isolignes, mais de valider un nuage de points independants.

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

## Ordre de validation

L'ordre correct est le suivant :

1. valider un nuage de points `(alpha, Mach)` representatifs ;
2. pour chaque point, verifier a la fois :
   - `c_i`
   - le statut de succes spectral / modal
   - la reconstruction du mode
3. seulement si un nombre suffisant de points est valide, revenir a une continuation locale ou a un balayage plus global ;
4. reconstruire les isolignes uniquement apres cette validation pointwise.

Autrement dit :

- on ne part plus du principe que "un point qui marche" implique "la grille marchera" ;
- on demande d'abord un ensemble de points validates en `c_i` et en mode.

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

## Condition avant les isolignes

Les isolignes supersoniques ne doivent etre lancees qu'apres avoir etabli un nuage de points de confiance avec :

- `best_success=True`
- `best_stage1_mismatch < 5e-2`
- `best_stage2_mismatch < 1e-2`
- une reconstruction modale visuellement credible
- des amplitudes aux bords faibles

Sans cette etape intermediaire, les isolignes risquent surtout de propager des erreurs d'amorcage ou de suivi de branche.

## Cible de reference pour le PINN supersonique

La reference supersonique ne doit pas etre pensee comme une carte modale continue et parfaite partout.

La cible utile pour le PINN est une reference hierarchisee :

- une base **gold** :
  - points modalement valides
  - eigenvaleur et mode juges fiables
- une base **silver** :
  - points spectraux credibles
  - mais sans validation modale suffisante
- une base **unresolved / branch ambiguous** :
  - points non fiables pour benchmarker le PINN

Pour le PINN, le comparatif doit ensuite etre fait ainsi :

- benchmark spectral dense sur la base `gold + silver`
- benchmark modal seulement sur la base `gold`

## Pourquoi une reference continue en `c_i` et en mode n'est pas possible partout

Dans le regime supersonique actuel, demander une courbe continue a la fois :

- en `c_i(alpha, Mach)`
- et en reconstruction modale

est trop fort pour la reference classique disponible.

Les raisons pratiques sont les suivantes :

- plusieurs familles de modes coexistent ;
- certaines branches supersoniques decroissent tres faiblement ;
- un point peut etre propre spectralement tout en echouant modalement ;
- des changements de branche apparaissent quand `alpha` ou `Mach` varient peu ;
- la troncature numerique du domaine et le choix de la branche asymptotique influencent fortement le raccord modal.

Consequence :

- une ligne continue en `c_i` peut etre construite beaucoup plus facilement qu'une ligne continue en modes valides ;
- si on force une reference modale continue, on risque de recoller artificiellement des points provenant de familles differentes ;
- cette "continuite" serait alors trompeuse pour le PINN.

La bonne reference n'est donc pas :

- une surface modale continue partout

mais :

- un nuage de points `gold`
- un squelette spectral `silver`
- et une carte explicite des zones ambigues.

## Base gold actuelle a conserver

Les points suivants doivent etre conserves comme ancrages `gold` actuels, car ils sont deja `validated` dans la reference modale fusionnee :

### `Mach = 1.2`

- `alpha = 0.150000`
- `alpha = 0.175000`
- `alpha = 0.187500`
- `alpha = 0.200000`
- `alpha = 0.208333`
- `alpha = 0.216667`

### `Mach = 1.3`

- `alpha = 0.100000`
- `alpha = 0.125000`
- `alpha = 0.150000`
- `alpha = 0.162500`
- `alpha = 0.175000`
- `alpha = 0.183333`
- `alpha = 0.191667`

### `Mach = 1.4`

- `alpha = 0.125000`
- `alpha = 0.137500`
- `alpha = 0.150000`
- `alpha = 0.162500`
- `alpha = 0.168750`

### `Mach = 1.5`

- `alpha = 0.125000`
- `alpha = 0.137500`
- `alpha = 0.150000`
- `alpha = 0.156250`
- `alpha = 0.162500`

Ces points sont la base modale de comparaison du futur PINN supersonique.

Note importante :

- `Mach = 1.4` est maintenant promu en base modale robuste apres correction de la lecture des isolignes de Blumen en `c_i`;
- l'ancienne estimation `blumen_ci` triait a tort certaines polylignes par `Mach`, ce qui sous-estimait artificiellement `c_i` autour de `M = 1.4`;
- la comparaison correcte se fait sur la branche principale de l'isoline digitalisee.

## Points candidats a confirmer avant promotion en gold

Ces points ont montre un signal utile, mais ils ne doivent pas encore etre promus automatiquement.

### `Mach = 1.4`

Zone suivante a auditer apres validation locale :

- `alpha = 0.175000`
- `alpha = 0.181250`
- `alpha = 0.187500`
- `alpha = 0.193750`
- `alpha = 0.200000`

### `Mach = 1.6`

Premiere cible pointwise a ouvrir :

- `alpha = 0.100000`
- `alpha = 0.125000`
- `alpha = 0.150000`
- `alpha = 0.175000`
- `alpha = 0.200000`

### Campagne suivante recommandee

Ordre pratique :

1. bande coeur :
   - `alpha in [0.10, 0.20]`
   - `Mach in [1.0, 1.8]`
2. bande elargie :
   - `alpha in [0.05, 0.25]`
   - `Mach in [1.0, 1.8]`

Scripts de soumission :

```bash
bash scripts/submit_supersonic_pointwise_core_band.sh
bash scripts/submit_supersonic_pointwise_extended_band.sh
```

Reglage prudent utilise pour ces campagnes :

- `MAX_Y_LIMIT = 800`
- `Y_LIMIT_FACTOR = 8`
- `MATCH_Y = 1.0`
- batch pointwise par Mach pour garder des temps de run lisibles
- `alpha = 0.162500`
- `alpha = 0.168750`

Zone de transition a traiter comme branche ambigue :

- `alpha = 0.175000`
- `alpha = 0.181250`
- `alpha = 0.187500`

### `Mach = 1.5`

Front a confirmer avant extension :

- `alpha = 0.165625`

Front ensuite a rechercher :

- `alpha = 0.168750`
- `alpha = 0.171875`
- `alpha = 0.175000`

## Points a chercher ensuite pour couvrir `Mach in [1.1, 1.8]`

L'objectif minimal pour le PINN est d'avoir des points spectraux et modaux eparpilles sur tout l'intervalle `Mach in [1.1, 1.8]`.

La liste suivante est l'ordre recommande.

### Priorite 1 : refermer les trous proches des zones deja stables

#### `Mach = 1.4`

- `alpha = 0.125000`
- `alpha = 0.137500`
- `alpha = 0.150000`
- `alpha = 0.162500`
- `alpha = 0.168750`

#### `Mach = 1.4` transition de branche

- `alpha = 0.172000`
- `alpha = 0.175000`
- `alpha = 0.178125`
- `alpha = 0.181250`
- `alpha = 0.184375`
- `alpha = 0.187500`

#### `Mach = 1.5` front modal

- `alpha = 0.165625`
- `alpha = 0.168750`
- `alpha = 0.171875`
- `alpha = 0.175000`

### Priorite 2 : ouvrir les Mach adjacents

#### `Mach = 1.1`

- `alpha = 0.150000`
- `alpha = 0.175000`
- `alpha = 0.200000`
- `alpha = 0.225000`

#### `Mach = 1.6`

- `alpha = 0.125000`
- `alpha = 0.150000`
- `alpha = 0.162500`
- `alpha = 0.175000`

### Priorite 3 : etendre plus loin en Mach

#### `Mach = 1.7`

- `alpha = 0.125000`
- `alpha = 0.150000`
- `alpha = 0.175000`

#### `Mach = 1.8`

- `alpha = 0.100000`
- `alpha = 0.125000`
- `alpha = 0.150000`

## Protocole exact a suivre

Le protocole recommande pour construire la reference supersonique utile au PINN est le suivant.

### Etape 0 - Figer la base gold

Ne plus toucher aux points `gold` actuels tant qu'un rerun ne les invalide pas explicitement.

Leur role :

- benchmark modal du futur PINN
- amorcage des nouvelles explorations

### Etape 1 - Reconfirmation branche guidee `M = 1.4`

Objectif :

- tester la branche intermediaire entre `M = 1.3` et `M = 1.5`
- sans la melanger aux autres familles

Commande :

```bash
ALPHAS=0.125000,0.137500,0.150000,0.162500,0.168750 \
OUTPUT_STEM=supersonic_shooting_point_batch_M140_branch_guided_reconfirm \
sbatch launch/jz_submit_supersonic_shooting_point_batch_M140_branch_guided.slurm
```

Promotion en `gold` seulement si :

- `best_status=validated`
- `best_stage1_mismatch < 5e-2`
- `best_stage2_mismatch < 1e-2`
- pas de soupcon de troncature de boite

### Etape 2 - Audit de la zone de transition `M = 1.4`

Objectif :

- ne pas forcer une continuite artificielle entre deux familles

Commande :

```bash
ALPHAS=0.172000,0.175000,0.178125,0.181250,0.184375,0.187500 \
INCLUDE_GENERIC_SEEDS=1 \
OUTPUT_STEM=supersonic_shooting_point_batch_M140_transition_dualseed \
sbatch launch/jz_submit_supersonic_shooting_point_batch_M140_branch_guided.slurm
```

Interpretation :

- si les points restent `spectral_only` ou changent brutalement de `c_r`, la zone reste `branch_ambiguous`
- on ne la promeut pas en `gold`

### Etape 3 - Extension locale du front `M = 1.5`

Objectif :

- pousser le front modal juste apres `alpha = 0.162500`

Commande :

```bash
POINTS="0.165625:1.50 0.168750:1.50 0.171875:1.50 0.175000:1.50" \
OUTPUT_STEM=supersonic_shooting_point_batch_M150_front_local \
sbatch launch/jz_submit_supersonic_shooting_point_batch.slurm
```

Le point `0.165625:1.50` est la premiere cible de promotion en `gold`.

### Etape 4 - Ouverture des Mach adjacents `1.1` et `1.6`

Objectif :

- obtenir des points modaux eparpilles plus tot sur l'intervalle `Mach in [1.1, 1.8]`
- sans attendre une surface complete

Commande :

```bash
POINTS="0.150000:1.10 0.175000:1.10 0.200000:1.10 0.225000:1.10 0.125000:1.60 0.150000:1.60 0.162500:1.60 0.175000:1.60" \
OUTPUT_STEM=supersonic_shooting_point_batch_M110_M160_seed_cloud \
sbatch launch/jz_submit_supersonic_shooting_point_batch.slurm
```

But :

- recuperer quelques points `validated`
- construire des ancres locales pour les Mach voisins

### Etape 5 - Extension vers `1.7` et `1.8`

Cette etape ne se lance qu'apres l'etape 4.

Commande :

```bash
POINTS="0.125000:1.70 0.150000:1.70 0.175000:1.70 0.100000:1.80 0.125000:1.80 0.150000:1.80" \
OUTPUT_STEM=supersonic_shooting_point_batch_M170_M180_seed_cloud \
sbatch launch/jz_submit_supersonic_shooting_point_batch.slurm
```

### Etape 6 - Densification locale uniquement apres succes pointwise

On ne relance une densification locale en `alpha` que si les points pointwise precedents donnent deja un petit nuage `validated` au Mach concerne.

### Etape 7 - Surface 2D seulement a la fin

Le tracking de surface modale 2D ne revient qu'apres avoir obtenu :

- une base `gold` modale sparse mais robuste
- une base `silver` spectrale plus dense
- une carte explicite des zones ambigues

## Regle de promotion

Un point peut etre promu en `gold` seulement si :

- `best_status = validated`
- `best_spectral_success = True`
- `best_mode_success = True`
- `best_stage1_mismatch < 5e-2`
- `best_stage2_mismatch < 1e-2`
- `edge_amp_fraction_max < 5e-2`
- le point reste stable sur au moins un rerun ou un amorcage voisin

Sinon :

- `validated` mais fragile -> a surveiller
- `spectral_only` -> `silver`
- branche qui saute -> `branch_ambiguous`
- echec du mode -> non utilisable pour benchmark modal
