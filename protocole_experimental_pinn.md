# Protocole expérimental PINN

Ce document résume l'état du travail PINN, ce qui a déjà été essayé, ce qui semble acquis et ce qui reste à faire.

## PINN subsonique

### Objectif

- apprendre à prédire `c_i` et le mode subsonique
- commencer sur un cas contrôlé à `Mach` fixé
- ensuite étendre à un modèle dépendant de `(alpha, Mach)`

### Séparation entre baseline et tests annexes

Le pipeline qui fonctionne actuellement doit être conservé comme baseline de travail.

Cette baseline sert à :

- produire les références PINN courantes ;
- comparer les nouvelles variantes ;
- éviter qu'un test méthodologique dégrade le protocole opérationnel.

En parallèle, les nouvelles idées sont testées comme **branches annexes**.

Leur objectif n'est pas de remplacer immédiatement le pipeline courant, mais de réduire progressivement la dépendance au solveur classique dans l'apprentissage.

La règle méthodologique est donc :

- on garde le pipeline actuel tant qu'il est le plus fiable ;
- on teste séparément les variantes à supervision classique réduite ;
- on ne les promeut que si elles restent compétitives face à la baseline.

### Direction stratégique

L'objectif long terme n'est pas d'optimiser indéfiniment un cas où le solveur classique est disponible.

L'objectif est de préparer des cas plus complexes où :

- la référence classique peut être coûteuse ;
- la référence classique peut être partielle ;
- ou la référence classique peut être absente.

Dans cette perspective, la bonne trajectoire est :

1. supervision classique dense pour verrouiller une baseline ;
2. supervision classique clairsemée pour entraîner un guide spectral ;
3. PINN principal contraint par ce guide, avec moins de classique dans la loss ;
4. à terme, transfert de ce schéma à des problèmes où le classique n'est plus disponible localement.

### Variante annexe `guide-window sparse`

La première variante explicitement alignée avec cet objectif est un schéma spectral en deux niveaux :

- un guide spectral appris sur un petit nombre d'ancres classiques en `c_i(alpha)` ;
- un PINN principal qui n'utilise plus la courbe classique dense dans sa loss, mais une information issue du guide.

Le rôle du guide est de fournir :

- une courbe centrale `mu(alpha)` ;
- une largeur de fenêtre `sigma(alpha)`.

Le rôle du PINN principal est ensuite :

- d'apprendre le spectral et le modal avec les losses physiques usuelles ;
- tout en restant proche de `mu(alpha)` ;
- et en évitant de sortir de la fenêtre `mu(alpha) ± sigma(alpha)`.

Dans ce schéma :

- le solveur classique dense n'est plus utilisé pour superviser directement `c_i` dans le réseau principal ;
- il n'est utilisé que pour construire le guide clairsemé et pour l'évaluation finale.

### Paramétrisation actuelle du PINN

Le cadre de travail actuel sépare déjà explicitement le spectral et le modal.

En 1D à `Mach` fixé, le modèle comporte :

- une tête spectrale `alpha -> c_i(alpha)`
- une tête modale `(xi, alpha) -> mode`

Dans les runs Riccati, la tête modale ne sort pas directement `rho, u, v, p`. Elle sort :

- `kappa(xi, alpha)`
- `q(xi, alpha)`

et on reconstruit ensuite

- `gamma = kappa + i q = p_y / p`
- puis `p`
- puis `rho, u, v`

Cette hiérarchie est importante :

- le bon objet primaire à apprendre est `p`, ou mieux `gamma`
- `u` et `v` sont des quantités reconstruites et mal conditionnées à bas `alpha`
- il ne faut donc pas les traiter comme sorties de base du réseau

### Comment la supervision de `c_i` est injectée

La supervision de `c_i` est un terme de perte explicite.

Concrètement, à chaque epoch :

- on tire un batch de valeurs de `alpha`
- sur ces `alpha`, on interpole la référence classique subsonique
- on compare `c_i` prédit par le réseau à `c_i` classique

Le terme ajouté à la loss est de la forme :

- `loss_ci = mean((c_i^pred - c_i^ref)^2)`

Donc oui : pour chaque point de supervision, la valeur classique est injectée comme cible scalaire dans la loss.

Important :

- à `Mach` fixé, un point de supervision signifie un **point en `alpha`**
- dans un futur modèle 2D, un point de supervision signifiera un **couple `(alpha, Mach)`**

### Ce que signifient les “8 points de supervision”

Le budget “8 points” retenu dans plusieurs runs 1D ne veut pas dire :

- 8 points fixes sur tout le plan `(alpha, Mach)`

Il veut dire :

- `n_alpha_supervision = 8`
- donc 8 valeurs de `alpha` sont envoyées dans le terme `loss_ci` à chaque epoch

Ces 8 points ne sont pas forcément les mêmes d'une epoch à l'autre.

Le sampling est adaptatif :

- une partie des `alpha` est tirée uniformément sur la bande
- une autre partie est tirée autour des `alpha` où le modèle est mauvais

Le but est de concentrer progressivement la supervision sur les zones difficiles sans avoir à superviser toute la bande densément.

### Forme actuelle de la loss

La loss totale est une somme de blocs de nature différente.

#### Bloc spectral

- supervision classique de `c_i`
- régularisation éventuelle de `c_i`
- lissage éventuel
- contrôle près de la neutralité si activé

#### Bloc physique modal

Pour les runs Riccati, la loss impose :

- le résidu de l'équation de Riccati
- les conditions asymptotiques aux bords
- les contraintes de bande proche des bords
- les contraintes de normalisation
- les contraintes de phase

Autrement dit, le réseau n'apprend pas librement un champ arbitraire :

- il apprend un champ `gamma`
- qui doit satisfaire l'équation physique et ses asymptotiques

#### Bloc classique modal

Quand il est activé, on ajoute des termes de guidage classique :

- ancres sur quelques `alpha`
- supervision de `q`
- supervision du mode `p`
- ou supervision complète `rho, u, v, p`

Depuis le dernier patch méthodologique, la référence modale utilisée pour ces termes n'est plus GEP-based :

- elle est désormais reconstruite à partir du **shooting subsonique**

Ce point est important, car il aligne enfin la loss modale du PINN sur la référence classique que l'on juge la plus crédible.

### Ce que résout réellement la formulation Riccati

La formulation Riccati actuelle est déjà inspirée du shooting, mais elle n'est pas encore un vrai “neural shooting”.

Aujourd'hui :

- le réseau prédit un champ global `kappa, q` sur tout le domaine
- la loss lui impose la Riccati, les asymptotiques et un mismatch de type shooting

Donc :

- la physique du shooting est déjà présente dans la loss
- mais pas encore dans l'architecture elle-même

Ce n'est pas encore :

- une branche gauche
- une branche droite
- un raccord central explicite appris

### Ansatz cible à moyen terme

L'architecture cible pour débloquer complètement la reconstruction modale doit être plus proche du shooting.

Le schéma visé est :

- une tête spectrale :
  - `c_i(alpha)` en 1D
  - puis `c_i(alpha, Mach)` en 2D subsonique
- une branche modale gauche
- une branche modale droite
- une variable de raccord amplitude/phase au centre

Dans cette logique :

- les asymptotiques sont imposées dans la paramétrisation autant que possible
- le réseau apprend surtout la correction et le raccord
- les champs `u` et `v` sont reconstruits à la fin, pas appris directement

### Position méthodologique sur le gel de `c_i`

Geler `c_i` à bas `alpha` est pertinent comme expérience de réparation, mais pas comme solution finale.

Ce gel sert à répondre à une seule question :

- peut-on réparer les modes à bas `alpha` sans casser une courbe `c_i` déjà bonne ?

En revanche, le modèle final visé ne doit pas rester découpé en deux objets indépendants figés.

La trajectoire cible est :

1. apprendre proprement le spectral
2. apprendre le modal avec `c_i` quasi figé
3. dégeler et faire un fine-tuning joint

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
- la meilleure base 1D actuelle pour `c_i` est maintenant [frozen_riccati_multibranch_ci_best](/Users/emma.grospellier/Thèse/These_PINN_KH_RT/assets/pinn_subsonic/mach_fixed/frozen_riccati_multibranch_ci_best)
- cette base est figée pour `c_i`, mais pas encore pour les modes

### Cas figé retenu à `Mach` fixé

Le cas de référence retenu pour fermer la phase `Mach` fixé est :

- protocole `classic_two_stage_repair`
- checkpoint retenu : `stage_b_pressure_overlap/model_best.pt`

Lecture méthodologique :

- le stage `balanced full-mode` seul améliore peu
- le gain utile vient du stage `pressure + balanced overlap`
- le refinément ultérieur améliore légèrement `p_rel`, mais reperd trop sur `peak`

Conclusion :

- le run figé est le meilleur compromis global
- le refinément n'est conservé que comme test de sensibilité, pas comme référence principale

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

- figer la base 1D `c_i` avec `riccati_multibranch`
- ne pas surpromettre la reconstruction modale tant que les heatmaps d'erreur de mode ne passent pas sous `5%`
- utiliser le classique comme vérité terrain explicite pour amplitude, phase et reconstruction des vitesses

### Conclusion 1D actuelle

Le point méthodologique important est maintenant clair :

- la reconstruction de `u` et `v` devait être corrigée en utilisant directement `p_y = gamma p` pour les runs Riccati
- après correction, `riccati_multibranch` devient clairement le meilleur candidat 1D
- `c_i` est suffisamment bon pour être figé
- les modes ne sont bons qu'à grand `alpha`, typiquement pour `alpha > 0.5`
- le verrou restant est donc localisé sur les petits `alpha`

En moyenne pour `riccati_multibranch` :

- pour `alpha <= 0.5` :
  - `p_rel_mean ≈ 0.155`
  - `u_rel_mean ≈ 0.295`
  - `v_rel_mean ≈ 0.281`
- pour `alpha > 0.5` :
  - `p_rel_mean ≈ 0.026`
  - `u_rel_mean ≈ 0.114`
  - `v_rel_mean ≈ 0.061`

Donc :

- la physique modale est presque correctement apprise à grand `alpha`
- la difficulté restante est concentrée à bas `alpha`
- la suite doit rester en 1D jusqu'à validation modale complète

### Prochaine étape immédiate

Avant tout sweep 2D ou tout passage au supersonique, la priorité est désormais :

- rester en 1D à `Mach` fixé
- conserver `riccati_multibranch` comme base `c_i`
- améliorer explicitement la reconstruction des modes pour `alpha <= 0.5`
- n'accepter le gel complet du 1D que si les heatmaps modales passent sous `5%`

### Protocole suivant : réparation modale bas `alpha` avec supervision structurelle sparse

Le résultat obtenu avec la cache modale shooting fixe désormais la stratégie :

- ne pas basculer vers une supervision full-mode dense
- conserver une physique majoritaire
- ajouter seulement un guidage parcimonieux sur les quantités les mieux conditionnées

Le point important est le suivant :

- la cache shooting améliore fortement `p`, `v` et la phase
- `u` reste la quantité la plus fragile
- le verrou n'est donc plus global, mais concentré sur la structure modale bas `alpha`

#### Principe général

La règle retenue pour la suite est :

- superviser **mieux**
- pas superviser **plus**

Cela signifie :

- ne pas imposer `rho, u, v, p` partout
- renforcer les contraintes sur `gamma = p_y / p`
- renforcer les asymptotiques et le raccord
- ne toucher aux vitesses que sous forme très localisée et rescalée si nécessaire

#### Quantités à privilégier

Ordre de priorité des quantités à guider :

1. `c_i`
2. `q = Im(gamma)`
3. `kappa = Re(gamma)`
4. `p`
5. `alpha u` et `alpha v` en dernier recours

Ce choix est motivé par le conditionnement du problème :

- `p` et `gamma` restent bien conditionnés
- `u` et `v` deviennent rapidement instables à bas `alpha`
- une supervision brute sur `u` et `v` risquerait de transformer le PINN en simple régression de champ

#### Alphas d'ancrage recommandés

Pour le guidage bas `alpha`, on retient une grille sparse fixe :

- `alpha = 0.05`
- `alpha = 0.10`
- `alpha = 0.15`
- `alpha = 0.20`
- `alpha = 0.30`
- `alpha = 0.50`

Rôle de cette liste :

- couvrir la zone franchement difficile
- garder un point à la frontière haute du régime difficile
- éviter une supervision dense sur toute la bande

#### Bloc de losses à conserver

Les termes déjà validés et à conserver sont :

- résidu Riccati
- conditions asymptotiques
- `riccati_boundary_band_kappa`
- `riccati_boundary_band_q`
- `riccati_shooting_match`
- normalisation
- fixation de phase
- supervision légère de `q`
- ancrage Riccati sparse

#### Bloc de losses à renforcer

Premier niveau de renforcement, sans changer l'architecture :

- `w_q_supervision`
- `w_riccati_anchor`
- `w_riccati_boundary_band_kappa`
- `w_riccati_boundary_band_q`
- `w_riccati_shooting_match`

Poids de départ recommandés après le run shooting-cache :

- `w_q_supervision = 5`
- `w_riccati_anchor = 2`
- `w_riccati_boundary_band_kappa = 5`
- `w_riccati_boundary_band_q = 10`
- `w_riccati_shooting_match = 12`

Ces valeurs ne sont plus des hypothèses abstraites :

- elles ont déjà montré un gain réel à bas `alpha`
- elles constituent donc le point de départ de référence

#### Extension minimale recommandée

Si ce premier bloc ne suffit pas, l'extension minimale ne doit pas être une supervision full-mode dense.

L'extension suivante à tester est :

- ancrage sparse sur `gamma` lui-même

L'idée est :

- comparer directement `kappa` et `q` reconstruits au classique
- sur peu d'alphas
- sur peu de points en `y`
- avec priorité donnée aux zones proches des bords et au voisinage du centre

Cette extension reste compatible avec l'esprit PINN :

- on guide la variable physique principale
- on n'impose pas un champ complet partout

#### Vitesses : politique retenue

Pour `u` et `v`, la règle méthodologique est :

- pas de supervision brute en premier recours
- pas de supervision dense plein domaine

Si une aide supplémentaire devient nécessaire, elle devra prendre la forme :

- d'une supervision sparse sur `alpha u`
- et éventuellement sur `alpha v`

L'objectif est de neutraliser une partie du mauvais conditionnement bas `alpha` sans casser le cadre physique du PINN.

#### Ordre des stages recommandé

Le prochain protocole ne doit pas réentraîner tout d'un bloc.

Ordre recommandé :

1. **Stage A**
- garder `c_i` gelé
- réparer le mode avec la cache shooting et les losses Riccati renforcées

2. **Stage B**
- conserver les mêmes alphas d'ancrage
- ajouter, si nécessaire, une ancre sparse sur `gamma`

3. **Stage C**
- uniquement si besoin
- ajouter une très légère supervision rescalée sur `alpha u` / `alpha v`

4. **Stage D**
- une fois la structure modale stabilisée
- dégeler doucement `c_i`
- faire un fine-tuning joint

#### Critère de validation

Le protocole ne sera considéré comme validé que si :

- `c_i` reste au niveau actuel
- `p_rel_mean < 5%`
- `u_rel_mean < 5%`
- `v_rel_mean < 5%`
- sur les heatmaps, pas seulement sur quelques overlays

En pratique, un progrès partiel est déjà jugé utile si :

- `p` descend sous `10%`
- `v` descend nettement
- et `u` devient le seul verrou résiduel clairement identifié

#### Décision méthodologique si le protocole échoue encore

Si cette stratégie sparse sur `gamma/q/raccord` ne suffit pas, alors le bon prochain pas ne sera pas :

- plus de supervision dense

mais :

- un changement d'architecture vers une version explicitement shooting-like
- avec branche gauche
- branche droite
- et raccord central appris

### Ce qu'il reste à faire ensuite

- ne plus prioriser la branche `first_order_real`
- concentrer les essais sur la branche Riccati multibranch
- imposer une reconstruction plus proche du shooting
- mieux contrôler les conditions de bord, les queues et le raccord central
- introduire si nécessaire une supervision modale légère et ciblée à bas `alpha`
- une fois le cas `Mach` fixé satisfaisant pour `c_i` et les modes, seulement alors étendre le modèle à `(alpha, Mach)`

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
- ce n'est pas un retard méthodologique : la priorité passe d'abord par le sweep subsonique `(alpha, Mach)` après gel des références classiques

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
