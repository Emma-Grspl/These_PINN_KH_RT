# Protocole expérimental PINN

Ce document résume l'état du travail PINN, ce qui a déjà été essayé, ce qui semble acquis et ce qui reste à faire.

## PINN subsonique

### Objectif

- apprendre à prédire `c_i` et le mode subsonique
- commencer sur un cas contrôlé à `Mach` fixé
- ensuite étendre à un modèle dépendant de `(alpha, Mach)`

### Ce qui a déjà été fait

- mise en place d'un baseline subsonique à `Mach = 0.5` avec sweep en `alpha`
- recherche externe sur `c_i` pour comprendre la difficulté d'identification purement physique
- calibration du nombre minimal de points classiques nécessaires pour superviser légèrement `c_i`
- comparaison explicite entre cas hybride et physique pure
- génération de plots de budget `c_i`, de heatmaps d'erreur et de comparaisons PINN vs classique
- mise en place d'audits de reconstruction modale et de courbes d'erreur de mode en fonction de `alpha`
- rédaction d'une note méthodologique sur la supervision légère du mode dans [NOTE_supervision_legere_mode.md](/Users/emma.grospellier/Thèse/These_PINN_KH_RT/subsonique_pinn/NOTE_supervision_legere_mode.md)

### Résultats déjà acquis

- pour `c_i`, une supervision légère est justifiée
- le budget de supervision minimal retenu actuellement est de 8 points classiques
- le cas physique pur pour `c_i` échoue clairement par rapport au cas hybride
- la baseline hybride 8 points constitue aujourd'hui la meilleure base de travail pour le subsonique

### Ce qui a été testé pour le mode

- formulation amplitude/phase
- comparaison mode PINN vs mode classique sur plusieurs `alpha`
- run `modefocus_lowalpha` pour renforcer l'apprentissage des bas `alpha`
- formulation premier ordre réel
- formulation premier ordre réel stabilisée
- préparation de variantes inspirées du système de Riccati

### Ce que les expériences ont montré

- le verrou principal n'est plus `c_i`, mais le mode
- l'amplitude est mauvaise surtout à bas `alpha`, approximativement pour `alpha < 0.45`
- la phase reste mauvaise sur une plage encore plus large, approximativement jusqu'à `alpha < 0.6`
- le run `modefocus_lowalpha` améliore un peu la phase
- cette amélioration se paie par une dégradation partielle de l'amplitude
- une faible erreur scalaire ne suffit pas à conclure que le mode est bien reconstruit
- les formulations `first_order_real` et `first_order_real_stabilized` peuvent retrouver un bon `c_i`, mais restent très mauvaises pour le mode

### Cas `first_order_real_stabilized`

Le run `first_order_real_stabilized` a clarifié un point important :

- la stabilisation améliore le comportement numérique global
- `c_i` redevient bon, avec un `best_ci_mae` de l'ordre de `4e-3`
- mais la reconstruction modale reste très mauvaise

En particulier :

- `p_rel` reste autour de `4.4`
- `env` reste autour de `4.4`
- la phase ne compense pas cette erreur structurelle

Donc :

- cette formulation ne résout pas le problème du mode
- elle produit encore un bon scalaire spectral avec une eigenfonction fausse
- elle ne constitue pas une base crédible pour la suite, sauf justification théorique nouvelle

### Pourquoi cette piste est abandonnée pour l'instant

La branche `first_order_real` est abandonnée à ce stade pour une raison simple :

- elle ne traite pas le vrai verrou expérimental, qui est la reconstruction du mode

Plus précisément :

- la version non stabilisée est numériquement trop instable
- la version stabilisée corrige surtout le comportement d'optimisation
- mais ne corrige pas l'identifiabilité de l'eigenfonction

Autrement dit :

- elle permet de retrouver un bon `c_i`
- sans forcer le réseau à converger vers le bon mode

Ce résultat est méthodologiquement suffisant pour arrêter d'investir sur cette piste dans l'immédiat :

- le problème n'est pas seulement un manque de stabilisation
- il est plus probablement lié au choix des variables, à la normalisation, aux conditions de bord et à la façon dont la loss identifie le mode

La branche `first_order_real` ne sera donc réouverte que si une motivation théorique nouvelle apparaît.

### Difficultés rencontrées

- la physique pure ne verrouille pas suffisamment le mode dans la formulation actuelle
- la phase est la partie la plus fragile
- amplitude et phase ne progressent pas forcément ensemble
- le découpage amplitude/phase peut introduire des ambiguïtés d'optimisation
- les formulations premier ordre réel testées jusqu'ici ont montré soit une instabilité forte, soit une stabilisation insuffisante pour corriger le mode

### Position méthodologique actuelle

- figer le baseline `c_i` avec 8 points de supervision légère
- ne pas surpromettre la reconstruction modale tant que les courbes d'erreur de mode ne sont pas bonnes
- utiliser le classique comme vérité terrain explicite pour amplitude et phase

### Ce qu'il reste à faire ensuite

- ne plus prioriser la branche `first_order_real`
- concentrer les essais sur les formulations Riccati, multibranch ou autres formulations mieux adaptées à la structure spectrale
- décider si la sortie réseau doit rester en amplitude/phase ou passer à une autre représentation réelle plus robuste
- mieux contrôler les conditions de bord et la normalisation modale
- introduire si nécessaire une supervision modale légère et ciblée, surtout à bas `alpha`
- une fois le cas `Mach` fixé satisfaisant, étendre le modèle à `(alpha, Mach)`

### Plan d'expériences local pour débloquer la reconstruction du mode

L'objectif immédiat n'est pas de lancer une grande campagne Jean Zay, mais d'identifier localement quelle formulation donne le meilleur compromis amplitude/phase à coût raisonnable.

Le principe retenu est :

- faire peu de runs
- sur peu de cas
- avec une comparaison propre entre formulations

#### Banc d'essai minimal

On fixe :

- `Mach = 0.5`
- supervision légère de `c_i` avec 8 points
- trois valeurs de `alpha` représentatives :
  - `alpha = 0.2`
  - `alpha = 0.5`
  - `alpha = 0.8`

Ce choix permet de séparer :

- la zone bas `alpha` où le mode échoue
- une zone intermédiaire
- une zone haut `alpha` où le comportement est meilleur

#### Vérité terrain utilisée

Pour chaque cas, on compare au classique sur :

- `c_i`
- amplitude
- phase
- éventuellement `Re(p)` et `Im(p)` si nécessaire

#### Métriques à suivre

Chaque formulation doit être évaluée avec les mêmes métriques :

- `ci_mae`
- erreur d'enveloppe
- erreur de phase
- erreur relative complexe globale sur `p`
- courbe d'erreur en fonction de `alpha`

Le critère de décision n'est pas la loss totale, mais :

- la qualité du mode à `c_i` comparable
- la stabilité entre runs
- l'amélioration à bas `alpha`

#### Ordre des tests

Il faut éviter de tout changer à la fois.

Ordre recommandé :

1. comparer les représentations de sortie
2. puis comparer les normalisations
3. puis comparer les conditions de bord

#### Bloc A : représentations à comparer

Premier bloc de tests :

- formulation actuelle amplitude/phase
- formulation `Re/Im`
- formulation `log-amplitude + phase`
- formulation Riccati ou premier ordre dérivé de `p'/p` si le coût d'implémentation reste raisonnable

Ici, tout le reste doit rester identique :

- même architecture
- même budget d'entraînement
- même supervision `c_i`
- mêmes poids de loss autant que possible

#### Bloc B : normalisations à comparer

Une fois la meilleure représentation identifiée, comparer plusieurs normalisations :

- ancre ponctuelle
- amplitude maximale normalisée à 1
- norme `L2` fixée
- phase fixée au point de pic

L'objectif est de réduire les degrés de liberté résiduels qui laissent la loss physique sous-déterminer le mode.

#### Bloc C : conditions de bord à comparer

Une fois la représentation et la normalisation fixées, comparer :

- BC faibles de type Dirichlet
- BC asymptotiques exponentielles
- BC sur la dérivée logarithmique

Le but est de voir si l'échec vient surtout de la représentation interne ou d'un mauvais verrouillage des bords.

#### Décision pratique

Une formulation sera retenue pour la suite si, à `c_i` comparable :

- elle améliore clairement l'amplitude
- elle améliore clairement la phase
- surtout à bas `alpha`
- sans dégrader fortement les cas moyens et hauts `alpha`

#### Pourquoi ces tests peuvent être faits en local

Ce protocole est volontairement léger :

- peu d'alphas
- `Mach` fixé
- comparaison ciblée

Il est donc adapté à des essais locaux, ce qui évite de dépendre trop tôt de Jean Zay tant que la bonne formulation n'est pas identifiée.

Jean Zay ne doit servir qu'après ce tri initial, pour :

- confirmer la meilleure formulation
- élargir la grille
- lancer les campagnes plus lourdes

## PINN supersonique

### Situation actuelle

- le travail PINN supersonique n'est pas encore réellement lancé
- ce n'est pas un retard méthodologique : la référence classique supersonique n'est pas encore verrouillée

### Pourquoi il ne faut pas aller trop vite

Le supersonique demande d'apprendre :

- `c_i`
- `c_r`
- le mode
- et surtout la bonne branche physique

Tant que le classique n'identifie pas de manière stable la bonne branche, entraîner un PINN supersonique risquerait surtout d'apprendre une mauvaise cible.

### Ce qui devra être fait une fois le classique verrouillé

- définir une base de données de référence `(alpha, Mach) -> (c_r, c_i, mode)`
- commencer par des cas simples :
  - point unique
  - sweep 1D en `alpha`
  - sweep 1D en `Mach`
- seulement ensuite passer au cas 2D `(alpha, Mach)`

### Questions de formulation déjà identifiées

- quelles sorties réseau prendre pour `c_r`, `c_i` et le mode
- quelle représentation choisir pour le mode :
  - amplitude/phase
  - variables réelles et imaginaires
  - système premier ordre réel
- quelles conditions de bord imposer
- quel niveau minimal de supervision légère accepter si la physique pure ne verrouille pas la branche

### Difficultés anticipées

- le supersonique cumule les difficultés spectrales du classique et les difficultés d'optimisation du PINN
- il faudra empêcher le réseau d'apprendre une mauvaise famille modale
- il faudra auditer séparément :
  - l'erreur sur `c_r`
  - l'erreur sur `c_i`
  - l'erreur sur le mode
  - la cohérence de branche

### Prochaine étape raisonnable côté PINN supersonique

- ne rien figer tant que le classique supersonique n'est pas proprement verrouillé
- préparer seulement les choix de formulation et les métriques d'audit
