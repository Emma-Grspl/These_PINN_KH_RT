# Protocole PINN subsonique `alpha-Mach`

Ce document fixe la prochaine étape PINN avant tout passage au supersonique.

## Objectif

- passer d'un PINN subsonique validé à `Mach` fixé vers un modèle dépendant de `(alpha, Mach)`
- reconstruire `c_i` sur une grille 2D
- reconstruire quelques modes représentatifs pour vérifier que le modèle reste utilisable hors de la ligne `Mach = 0.5`

## Précondition déjà satisfaite

Le cas `Mach` fixé avec sweep en `alpha` est figé dans :

- [assets/pinn_subsonic/frozen_fixed_mach_alpha_sweep_best](/Users/emma.grospellier/Thèse/These_PINN_KH_RT/assets/pinn_subsonic/frozen_fixed_mach_alpha_sweep_best)

Ce cas sert maintenant de baseline stable. Il ne faut plus le retoucher avant d'avoir évalué le sweep 2D.

## Question scientifique

Le point à trancher est simple :

- le PINN subsonique est-il capable de sortir un `c_i` crédible et un mode exploitable pour un couple `(alpha, Mach)` arbitraire, et pas seulement le long d'une ligne `Mach` fixée ?

## Référence classique à utiliser

La vérité terrain doit venir du workflow classique subsonique déjà retenu :

- [assets/classic_subsonic/data/subsonic_hybrid_growth_map.csv](/Users/emma.grospellier/Thèse/These_PINN_KH_RT/assets/classic_subsonic/data/subsonic_hybrid_growth_map.csv)
- [assets/classic_subsonic/data/subsonic_hybrid_error_summary.json](/Users/emma.grospellier/Thèse/These_PINN_KH_RT/assets/classic_subsonic/data/subsonic_hybrid_error_summary.json)

Pour les modes, deux niveaux suffisent :

- vérité complète sur quelques points représentatifs
- pas de reconstruction modale exhaustive sur toute la grille dans le premier sweep

## Sorties attendues

À la fin, on veut exactement :

- une table `c_i(alpha, Mach)` classique
- une table `c_i(alpha, Mach)` PINN
- une heatmap d'erreur sur `c_i`
- une figure d'isolignes `c_i` PINN vs classique
- une figure d'erreur superposée aux isolignes
- quelques planches de modes `rho, u, v, p` sur des points `(alpha, Mach)` choisis

Noms d'artefacts recommandés :

- `subsonic_pinn_alphamach_classic_ci_surface.csv`
- `subsonic_pinn_alphamach_pinn_ci_surface.csv`
- `subsonic_pinn_alphamach_ci_error_surface.csv`
- `subsonic_pinn_alphamach_ci_error_heatmap.png`
- `subsonic_pinn_alphamach_ci_isolines_overlay.png`
- `subsonic_pinn_alphamach_ci_isolines_with_error.png`
- `subsonic_pinn_alphamach_mode_points.csv`
- `subsonic_pinn_alphamach_modes.pdf`

## Maillage expérimental recommandé

Premier sweep recommandé :

- une grille modérée en `alpha`
- une grille modérée en `Mach`
- assez dense pour voir les isolignes
- pas trop dense pour pouvoir itérer vite

Règle pratique :

- commencer par une grille de validation intermédiaire
- ne densifier qu'une fois les premières heatmaps lues

Les points de modes ne doivent pas être pris au hasard. Il faut au minimum :

- un point au centre de bande
- un point bas `alpha`
- un point haut `alpha`
- un point où `Mach` diffère nettement de `0.5`

## Protocole d'entraînement

### 0. Référence classique

Avant tout entraînement PINN :

- fixer la grille `(alpha, Mach)` classique
- calculer ou consolider `c_i` sur cette grille
- choisir les points modaux qui serviront à l'audit qualitatif

Ce point est indispensable. Le sweep PINN ne doit pas être lancé sans une grille classique claire de comparaison.

### 1. Baseline

- reprendre l'architecture et les normalisations du meilleur cas `Mach` fixé
- injecter `(alpha, Mach)` comme paramètres d'entrée explicites
- garder `c_i` comme cible scalaire principale

### 2. Cible principale

- d'abord verrouiller `c_i`
- ensuite seulement auditer les modes

Il ne faut pas inverser cet ordre. Une belle planche de mode n'a pas de valeur si le `c_i` est faux sur la carte 2D.

### 3. Cible modale

Pour ce premier sweep 2D :

- ne pas imposer une supervision modale lourde partout
- faire un audit localisé sur quelques couples `(alpha, Mach)`
- garder la logique par régimes déjà observée sur le cas `Mach` fixé

## Métriques à produire

### Pour `c_i`

- MAE absolue
- erreur relative
- heatmap 2D
- isolignes superposées PINN vs classique

### Pour les modes

- `p_rel`
- erreur d'enveloppe
- erreur de phase
- erreur de pic / localisation du maximum

Ces métriques doivent être sorties point par point pour les cas modaux choisis.

## Figures à produire

Figure 1 :

- `c_i(alpha, Mach)` classique

Figure 2 :

- `c_i(alpha, Mach)` PINN

Figure 3 :

- heatmap d'erreur sur `c_i`

Figure 4 :

- isolignes `c_i` classique vs PINN

Figure 5 :

- isolignes `c_i` avec carte d'erreur en arrière-plan

Figure 6 :

- quelques modes `rho, u, v, p` pour des points `(alpha, Mach)` représentatifs

## Critère de réussite

Le sweep sera considéré comme suffisamment bon si :

- les isolignes `c_i` sont cohérentes globalement
- la heatmap d'erreur reste lisible et localisée
- les points modaux choisis restent exploitables hors du seul cas `Mach = 0.5`

## Ce qu'on ne demande pas encore

À ce stade, on ne demande pas :

- une perfection uniforme du mode sur toute la grille
- un passage immédiat au supersonique
- une reconstruction exhaustive de tous les champs sur tout `(alpha, Mach)`

Le but est de valider la généralisation subsonique 2D avant de changer de régime physique.
