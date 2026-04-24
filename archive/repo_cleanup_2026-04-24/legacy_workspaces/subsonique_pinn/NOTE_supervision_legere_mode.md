## Pourquoi une supervision légère reste justifiée pour le mode

### Point de départ

Pour `c_i`, le besoin d'hybridation est assez clair :

- `c_i` est un scalaire spectral
- la loss physique laisse subsister plusieurs candidats compatibles
- quelques points classiques suffisent à lever cette ambiguïté

Pour le **mode**, l'intuition est moins immédiate, parce qu'en théorie il est gouverné par la même physique que le solveur classique. On pourrait donc s'attendre à ce qu'une loss purement physique suffise.

Les résultats montrent que ce n'est pas le cas dans l'état actuel de la formulation.

### Constat expérimental

Sur le sweep subsonique à `M = 0.5` :

- la supervision légère de `c_i` corrige très fortement la valeur propre
- mais la reconstruction modale reste inhomogène en `alpha`

En pratique :

- l'amplitude du mode reste mauvaise pour `alpha < 0.45`
- la phase reste mauvaise plus longtemps, jusqu'à environ `alpha < 0.60`
- un entraînement "mode-focus low-alpha" améliore un peu la phase
- mais cette amélioration se paye par une dégradation partielle de l'amplitude

Donc :

- la physique seule ne suffit pas à reconstruire proprement le mode sur tout le domaine
- et un simple reweighting de la loss ne règle pas complètement le problème

### Pourquoi la physique pure a du mal sur le mode

Le point important est le suivant :

- le solveur classique résout un problème spectral discrétisé, très rigide
- le PINN, lui, apprend une représentation fonctionnelle sous contraintes

Ces deux objets ne sont pas numériquement équivalents, même s'ils reposent sur la même équation.

Le mode est plus difficile que `c_i` pour plusieurs raisons :

1. le mode est une **fonction complexe**, pas un scalaire

- il faut reconstruire correctement l'enveloppe
- mais aussi la phase
- et la phase est précisément la partie la plus fragile

2. la loss physique n'identifie pas toujours fortement l'eigenfonction

- plusieurs profils peuvent satisfaire à peu près la même équation
- tout en donnant des formes modales différentes
- surtout dans la zone bas-`alpha`

3. le problème est mal conditionné selon `alpha`

- à grand `alpha`, le mode est plus facile à verrouiller
- à bas `alpha`, l'amplitude et surtout la phase deviennent beaucoup plus ambiguës

4. l'optimisation PINN introduit de la liberté fonctionnelle

- la représentation neurale peut produire des profils qui ont un résidu faible
- sans pour autant reproduire fidèlement le mode classique

Autrement dit :

- la physique pure impose bien une structure
- mais elle ne ferme pas complètement l'espace des solutions admissibles pour le mode

### Pourquoi la phase est le vrai verrou

Les essais récents montrent un schéma stable :

- l'amplitude peut être améliorée assez tôt
- la phase reste le composant le plus difficile

Le run `modefocus_lowalpha` le confirme :

- la phase progresse légèrement
- mais l'amplitude se dégrade un peu

Ce comportement est important, parce qu'il montre que :

- amplitude et phase ne sont pas automatiquement alignées dans la loss actuelle
- améliorer l'une peut dégrader l'autre

Donc le problème n'est pas simplement "mettre plus de poids sur la physique".
Le problème est que la loss actuelle ne cible pas assez directement la bonne information modale.

### Pourquoi une supervision légère du mode est méthodologiquement acceptable

Le point à défendre n'est pas :

- "la physique pure est inutile"

Le point à défendre est :

- "la physique pure fournit l'ossature du problème"
- "une supervision légère sert uniquement à lever l'ambiguïté résiduelle sur le mode"

Cette logique est exactement la même que pour `c_i`, mais appliquée à un objet plus riche.

L'idée n'est donc pas de superviser densément tout le champ.
L'idée est d'ajouter juste assez d'information classique pour :

- verrouiller l'enveloppe là où elle dérive
- contraindre la phase là où la loss physique est trop plate
- sans transformer le PINN en simple régression supervisée

### Ce qu'on cherche à éviter

Sans supervision légère du mode, on observe deux risques :

1. une reconstruction modale correcte seulement sur une partie du domaine
2. un compromis amplitude/phase mal placé, surtout à bas `alpha`

Cela donne un modèle qui :

- respecte globalement la physique
- retrouve bien `c_i`
- mais ne reconstruit pas de façon fiable l'eigenfonction complète

Or, si l'objectif scientifique est de revendiquer une reconstruction modale, ce n'est pas suffisant.

### Position méthodologique retenue

La position la plus propre est donc :

- garder une forte composante de physique dans la loss
- conserver une supervision très parcimonieuse sur `c_i`
- et accepter si nécessaire une supervision modale légère, ciblée sur les zones réellement ambiguës

En pratique, cela veut dire :

- ne pas superviser tout le mode partout
- cibler prioritairement les `alpha` bas
- et mesurer explicitement le nombre minimal d'ancres modales nécessaires

### Message scientifique défendable

Le message n'est pas :

- "le mode ne peut pas être reconstruit en physique pure"

Le message est :

- "dans cette formulation PINN, la physique pure ne verrouille pas suffisamment le mode sur tout le domaine, en particulier sa phase à bas `alpha`"
- "une supervision modale légère est donc justifiée comme correction minimale d'une ambiguïté résiduelle, pas comme substitution de la physique"

### Conséquence pratique

Avant d'ajouter une supervision modale dense, le bon protocole est :

1. identifier la zone où la physique pure échoue
2. ajouter quelques ancres classiques seulement dans cette zone
3. vérifier si un petit nombre de points suffit
4. documenter le budget minimal nécessaire

Le but n'est pas de "corriger" le PINN partout.
Le but est d'aider uniquement là où la loss physique actuelle reste sous-déterminée.
