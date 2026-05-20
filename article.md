# Plan d'article

Ce document sert de squelette pour un futur article. Il ne fige pas encore les resultats finaux. L'objectif est de cadrer :

- la question scientifique ;
- l'angle "physique numerique" ;
- le fil narratif entre solveurs classiques et PINNs ;
- la structure des sections, figures et messages principaux.

## 1. Positionnement general

### 1.1. Idee centrale

L'article doit montrer qu'un probleme spectral d'instabilite hydrodynamique compressible, pertinent pour la modelisation reduite des plasmas de fusion, peut etre traite par PINN a condition de :

- verrouiller une reference classique fiable ;
- distinguer clairement les regimes faciles et difficiles ;
- assumer que la physique seule ne selectionne pas toujours correctement la bonne branche modale ;
- introduire, quand c'est necessaire, un guidage classique leger ou fort.

### 1.2. Angle de publication

L'angle n'est pas :

- "les PINNs marchent partout" ;
- ni "on remplace les solveurs classiques".

L'angle plus solide est :

- un probleme spectral non trivial de physique des fluides compressibles ;
- une analyse methodique des limites des solveurs classiques et des PINNs ;
- un protocole hybride qui clarifie quand la physique seule suffit et quand elle ne suffit plus.

### 1.3. Public vise

- communaute "computational / numerical physics" ;
- lecteurs interesses par :
  - eigenvalue problems in hydrodynamic stability,
  - physics-informed learning,
  - hybrid numerical / ML workflows,
  - reduced modeling for fusion-relevant flows.

## 2. Titres de travail possibles

### Option A

`Physics-informed neural networks for compressible Kelvin-Helmholtz eigenmodes: limits of pure-physics training and hybrid guidance strategies`

### Option B

`Learning unstable modes of compressible shear layers with PINNs: a benchmark against classical shooting and generalized eigenvalue solvers`

### Option C

`From classical spectral solvers to PINNs for compressible Kelvin-Helmholtz instability: branch selection, modal reconstruction, and hybrid supervision`

## 3. Resume scientifique en 5 lignes

Une version courte du message de l'article pourrait etre :

1. On etudie un probleme spectral de type Kelvin-Helmholtz compressible dans une couche de cisaillement.
2. On construit d'abord une reference classique, en distinguant soigneusement solveur de tir et solveur GEP.
3. On montre que le PINN reproduit bien le coeur du spectre dans certains regimes, mais que les extremites en parametre posent un probleme de selection modale.
4. On montre ensuite qu'un guidage classique cible permet de reparer ces regimes difficiles.
5. L'apport principal n'est pas seulement un resultat de performance, mais une cartographie claire des regimes ou la physique informee seule est suffisante ou insuffisante.

## 4. Questions scientifiques a poser explicitement

L'article doit etre structure autour de questions nettes.

### Q1. Probleme physique

Peut-on reconstruire de maniere fiable les valeurs propres complexes et les modes propres d'une instabilite de cisaillement compressible sur une bande de parametres `(alpha, Mach)` ?

### Q2. Probleme numerique

Quels solveurs classiques fournissent une reference robuste pour ce probleme, et sur quelles quantites ?

### Q3. Probleme PINN

Un PINN contraint par la physique peut-il apprendre simultanement :

- la croissance `c_i`,
- la vitesse de phase `c_r`,
- et la structure modale ?

### Q4. Probleme methodologique

Quand la physique seule ne suffit pas, quelle forme minimale de supervision classique faut-il ajouter ?

## 5. Structure proposee de l'article

## 5.1. Introduction

### Role de cette section

Installer le contexte physique et numerique, puis poser la contribution.

### Points a couvrir

- Contexte large : instabilites hydrodynamiques et magnetohydrodynamiques dans les plasmas de fusion.
- Motivation "reduced models" :
  - comprendre les mecanismes de croissance lineaire ;
  - disposer de solveurs rapides pour scans parametriques ;
  - preparer des modeles data-driven / physics-informed.
- Interet des PINNs :
  - apprentissage avec contraintes physiques ;
  - potentiel pour les problemes spectraux parametres ;
  - mais difficulte de selection de branche et de reconstruction de modes.
- Limite de la litterature :
  - beaucoup de demonstrations PINN portent sur des PDE d'evolution ou des ODE simples ;
  - moins de travaux abordent des problemes spectraux complexes avec multi-branches et conditions asymptotiques.
- Message de l'article :
  - sur ce probleme, la difficulte centrale n'est pas seulement de predire un scalaire, mais de suivre la bonne famille modale.

### Dernier paragraphe de l'introduction

Il doit annoncer explicitement :

- le probleme etudie ;
- les solveurs classiques compares ;
- le PINN propose ;
- et la these principale :
  - la physique seule suffit dans certains regimes ;
  - un guidage classique devient necessaire dans les regimes difficiles.

## 5.2. Probleme physique

### Objectif

Expliquer la physique assez clairement pour un journal de physique numerique, sans noyer le lecteur.

### Sous-sections conseillees

#### 5.2.1. Couche de cisaillement compressible et instabilite KH

- definition du profil de base ;
- variable transverse `y`, nombre d'onde `alpha`, nombre de Mach `M` ;
- difference entre regime subsonique et supersonique ;
- interpretation de `c = c_r + i c_i`.

#### 5.2.2. Probleme spectral lineaire

- linearisation autour du profil de base ;
- recherche de modes normaux ;
- systeme aux valeurs propres ;
- interpretation physique :
  - `c_i > 0` : instable,
  - `c_r` : vitesse de phase,
  - vecteur propre : structure spatiale du mode.

#### 5.2.3. Reference a Blumen

- expliquer que Blumen sert de reference externe historique ;
- préciser ce qui est robuste dans cette comparaison et ce qui l'est moins ;
- introduire deja la difference de fiabilite entre `c_i` et `c_r` si besoin, mais sans anticiper tout le resultat.

### Figures possibles

- schema simple de la couche de cisaillement ;
- exemple de mode en subsonique et supersonique ;
- rappel visuel d'isolignes historiques type Blumen.

## 5.3. Solveurs classiques de reference

### Objectif

Montrer que l'etude PINN repose sur une reference numerique serieuse, pas sur une comparaison flottante.

### Sous-sections conseillees

#### 5.3.1. Solveur de tir

- idee generale :
  - integrer depuis les asymptotiques ;
  - raccorder les solutions ;
  - chercher `c_r, c_i` qui satisfont le probleme spectral.
- ingredients a decrire :
  - conditions asymptotiques,
  - taille de boite,
  - parametres `kappa`, `q`,
  - matching.

#### 5.3.2. Solveur GEP

- discretisation spatiale ;
- construction du probleme de valeurs propres generalise ;
- role du mapping, de la boite et du maillage ;
- selection des modes les plus instables.

#### 5.3.3. Audit des solveurs classiques

- pourquoi il faut une reference avant de juger le PINN ;
- ce qu'on compare :
  - `c_i`,
  - `c_r`,
  - structure modale,
  - continuite de branche.

### Message attendu

Cette section doit faire comprendre que :

- le classique n'est pas trivial non plus ;
- le shooting a ete retenu comme reference pratique la plus robuste ;
- le GEP reste utile comme outil d'audit et de compréhension, mais plus sensible numeriquement.

## 5.4. PINN : formulation et enjeux

### Objectif

Cette section doit etre plus detaillee que la physique, parce que c'est le coeur methodologique du papier.

### Sous-sections conseillees

#### 5.4.1. Rappel general sur les PINNs

- principe :
  - un reseau approxime la solution ;
  - les pertes viennent du residu de l'equation, des BC et de contraintes auxiliaires ;
  - il n'y a pas necessairement de supervision dense sur toute la solution.
- expliquer pourquoi c'est interessant pour un probleme spectral parametre.

#### 5.4.2. Parametrisation choisie ici

- entree du reseau :
  - `alpha`, eventuellement `Mach`, et variable spatiale ;
- sorties :
  - branche `c_i`,
  - branche modale,
  - eventuellement representation amplitude / phase ;
- raison du decouplage entre quantite scalaire et structure modale.

#### 5.4.3. Pertes physiques

- residu de l'equation ;
- conditions aux bords / asymptotiques ;
- normalisation du mode ;
- contraintes de phase ou de localisation si utilisees.

#### 5.4.4. Pourquoi les problemes spectraux sont plus difficiles que des PDE standard

- non unicite modale ;
- branches voisines ;
- perte physique satisfaite par plusieurs structures ;
- risque de bonne valeur propre mais mauvais mode ;
- compromis global lorsqu'on entraine sur une bande de parametres.

#### 5.4.5. Guidage classique

- definir clairement les differents niveaux de guidage :
  - guidage leger sur `c_i` ou `q`,
  - ancrage spatial,
  - supervision de mode partielle,
  - supervision full-mode.
- insister sur le fait que ce guidage n'est pas la methode par defaut, mais un outil controle pour les regimes difficiles.

### Message attendu

Le lecteur doit comprendre que le vrai sujet n'est pas juste "un PINN pour un probleme aux valeurs propres", mais :

- comment representer et entrainer un PINN quand la branche physique n'est pas automatiquement identifiable.

## 5.5. Protocole numerique

### Objectif

Fournir une section suffisamment reproductible, sans encore remplir tous les chiffres finaux.

### Sous-sections conseillees

#### 5.5.1. Domaine parametrique

- sous-sonique / sur-sonique ;
- `Mach` fixe puis balayage en `alpha` ;
- eventuelle extension vers `(alpha, Mach)`.

#### 5.5.2. Definition des references

- solveur classique retenu selon le regime ;
- donnees Blumen utilisees comme audit externe ;
- distinction entre quantites de calibration et quantites de validation.

#### 5.5.3. Metriques

- erreur sur `c_i` ;
- erreur sur `c_r` ;
- erreur modale `p_rel`, enveloppe, phase, position du pic ;
- overlap modal si retenu.

#### 5.5.4. Regimes de difficulte

Cette sous-section sera importante. Elle peut formaliser le decoupage :

- regime central en `alpha` ;
- bas `alpha` ;
- haut `alpha`.

### Message attendu

Le protocole doit montrer qu'on n'evalue pas le PINN "en moyenne" seulement, mais qu'on regarde explicitement les regimes qui posent des problemes de selection de branche.

## 5.6. Resultats

### Remarque

Cette section est a garder comme squelette pour l'instant.

### Structure conseillee

#### 5.6.1. Validation de la reference classique

- subsonique ;
- supersonique ;
- discussion de la robustesse relative de `c_i` et `c_r`.

#### 5.6.2. Resultats PINN en regime subsonique central

- montrer qu'en regime central, la physique seule ou quasi seule suffit.

#### 5.6.3. Regime bas-alpha

- montrer que la difficulte principale est l'identifiabilite du mode ;
- montrer le benefice d'un guidage classique leger.

#### 5.6.4. Regime haut-alpha

- montrer que le verrou n'est plus `c_i`, mais la famille modale ;
- comparer les niveaux de supervision testes.

#### 5.6.5. Resultats supersoniques

- selon l'avancement, soit comme resultats complets ;
- soit comme ouverture / section exploratoire.

## 5.7. Discussion

### Objectif

Interpreter les resultats, pas juste les repeter.

### Points a discuter

- pourquoi la physique seule marche dans certains regimes ;
- pourquoi elle echoue aux extremites ;
- pourquoi `c_i` est plus facile / plus robuste que le mode ;
- pourquoi `c_r` est la quantite la plus delicate a utiliser comme verite forte ;
- ce que cela dit sur les PINNs pour les problemes spectraux multi-branches.

### Message fort possible

La conclusion la plus interessante n'est peut-etre pas la performance brute du PINN, mais la mise en evidence d'une hiérarchie de difficultes :

- scalar instability growth,
- branch selection,
- modal shape reconstruction.

## 5.8. Conclusion

### A viser

Une conclusion courte en trois blocs :

1. ce qui est etabli ;
2. ce qui reste difficile ;
3. ce que cela ouvre pour la suite.

### Exemple de message final

- les solveurs classiques restent indispensables pour verrouiller les references ;
- les PINNs peuvent reproduire correctement une large partie du probleme ;
- mais les regimes de faible ou forte difficulte modale demandent une hybridation controlee.

## 5.9. Appendices possibles

- details de derivation du probleme spectral ;
- details des solveurs classiques ;
- details d'implementation du PINN ;
- choix des hyperparametres ;
- figures supplementaires de modes ;
- audit de sensibilite numerique.

## 6. Plan de figures

Ce plan peut servir tres tot, meme avant les resultats finaux.

### Figure 1

Schema du probleme physique :

- profil de base ;
- definition de `alpha`, `Mach`, `y` ;
- interpretation de `c_r`, `c_i`.

### Figure 2

Reference classique subsonique :

- carte de `c_i` ou coupe representative ;
- exemple de mode de reference.

### Figure 3

Reference classique supersonique :

- branche shooting ;
- comparaison a Blumen ;
- exemple modal.

### Figure 4

Architecture ou schema conceptuel du PINN :

- entrees ;
- sorties ;
- pertes ;
- interactions entre branche `c_i` et branche mode.

### Figure 5

Illustration des trois regimes en `alpha` :

- regime central ;
- bas `alpha` ;
- haut `alpha`.

### Figure 6

Exemple de succes du PINN en regime central.

### Figure 7

Exemple bas-alpha :

- echec pur physique ;
- reparation par guidage leger.

### Figure 8

Exemple haut-alpha :

- bon `c_i`, mauvais mode ;
- puis amelioration par supervision classique.

### Figure 9

Synthese des performances selon les regimes.

## 7. Plan de tableaux

### Tableau 1

Resume des solveurs classiques :

- shooting,
- GEP,
- domaine d'utilisation,
- points forts,
- limites.

### Tableau 2

Resume des variantes PINN :

- physique seule,
- guidage leger,
- supervision modale,
- full-mode,
- two-stage.

### Tableau 3

Erreurs quantitatives par regime :

- `c_i`,
- `c_r`,
- metriques modales.

## 8. Contributions possibles a annoncer

Il faudra choisir une formulation sobre. Une liste plausible :

1. formulation d'un benchmark spectral compressible pour PINNs, base sur une reference classique auditee ;
2. analyse comparative entre solveur de tir et GEP pour la reconstruction des branches instables ;
3. mise en evidence de regimes de difficulte distincts en `alpha` ;
4. proposition d'une strategie hybride graduelle pour la reconstruction PINN des modes propres ;
5. discussion de l'usage prudent de references historiques digitalisees, notamment pour `c_r`.

## 9. Ce qu'il vaut mieux ne pas promettre trop tot

Pour rester credible, il vaut mieux eviter de promettre trop vite :

- une couverture complete du supersonique si ce n'est pas verrouille ;
- une superiorite generale des PINNs sur les solveurs classiques ;
- une reconstruction parfaite de `c_r` dans toutes les zones ;
- une methode universelle independante du regime.

## 10. Message editorial a garder en tete

Si l'article doit passer dans un journal de physique numerique, il faut privilegier :

- un probleme bien pose ;
- des references classiques solides ;
- une analyse honnete des echecs ;
- une contribution methodologique claire ;
- et une discussion precise sur ce que les PINNs apportent, et sur ce qu'ils n'apportent pas encore.

Le papier sera probablement plus fort s'il raconte :

- comment un probleme spectral apparemment simple cache une vraie difficulte de selection modale,
- et comment cette difficulte force a concevoir un protocole hybride propre,

plutot que s'il essaie trop tot de vendre un "PINN general qui marche partout".
