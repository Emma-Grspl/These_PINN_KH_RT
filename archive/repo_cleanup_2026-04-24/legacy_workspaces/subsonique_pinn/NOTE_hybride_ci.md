## Pourquoi passer en hybride pour `c_i` et pas pour le mode

### Constat expérimental

Les essais récents montrent une séparation nette entre deux sous-problèmes :

- la **reconstruction du mode** peut être bonne en **physique pure**
- l'**identification de la valeur propre imaginaire `c_i`** ne l'est pas

En pratique :

- avec la formulation Riccati et les contraintes modales internes, le PINN reconstruit correctement la forme du mode
- en revanche, plusieurs valeurs de `c_i` donnent des modes visuellement et quantitativement proches
- la loss physique du mode ne sélectionne donc pas assez fortement la bonne valeur propre

Le point important est que ce n'est pas un échec global de la physique pure. C'est un échec **spectral**, pas un échec **modal**.

### Pourquoi `c_i` est plus difficile que le mode

Le mode est contraint par :

- l'équation différentielle locale
- les conditions aux bords
- les contraintes de symétrie / centrage / décroissance
- la structure géométrique de la solution

`c_i`, lui, est un **scalaire spectral**. Il est beaucoup moins directement contraint par les seules pertes locales. En conséquence :

- le paysage de loss reste relativement plat en `c_i`
- plusieurs candidats en `c_i` restent compatibles avec un bon mode
- le PINN peut donc reconstruire le bon mode sans identifier correctement la bonne croissance

Autrement dit :

- la physique pure suffit à apprendre la structure du mode
- elle ne suffit pas, sous la forme actuelle, à verrouiller la bonne valeur propre

### Pourquoi l'hybride est raisonnable

Le compromis proposé est le suivant :

- **mode** appris en physique pure
- **`c_i`** aidé par un petit nombre de points issus du solveur classique

Cette stratégie est cohérente parce qu'elle respecte la séparation observée expérimentalement :

- on ne remet pas de supervision classique sur la partie que le PINN sait déjà apprendre seul
- on n'utilise l'information classique que là où elle est réellement nécessaire

L'idée n'est donc pas de revenir à un modèle entièrement supervisé, mais de construire un **hybride minimal** :

- le solveur classique sert seulement d'ancrage spectral
- le PINN garde la responsabilité de la reconstruction du mode

### Pourquoi ne pas insister en physique pure pour `c_i`

On a déjà testé plusieurs directions en physique pure :

- apprentissage joint de `c_i`
- `c_i` scalaire
- contraintes spectrales globales
- matching Riccati embarqué
- recherche externe sur `c_i`

Le diagnostic est stable :

- la bonne zone en `c_i` peut parfois être retrouvée par recherche externe
- mais cela devient coûteux et fragile dès qu'on veut passer à un sweep en `alpha`
- une fenêtre fixe en `c_i` ne se généralise pas sur tout le domaine

Le passage en hybride pour `c_i` n'est donc pas un abandon prématuré de la physique pure. C'est une réponse pragmatique au fait que, dans l'état actuel de la formulation, la partie spectrale est sous-déterminée.

### Position méthodologique retenue

La position proposée est :

- conserver comme résultat fort que **la reconstruction du mode est obtenue en physique pure**
- accepter qu'une **aide classique minimale** soit nécessaire pour `c_i`
- documenter explicitement le **nombre minimal de points classiques** requis pour retrouver de bons résultats

### Ce qu'il faut documenter maintenant

Le bon objectif n'est plus "pur ou hybride" au sens binaire. Le bon objectif est :

- **quel est le nombre minimal de points classiques sur `c_i` qui suffit ?**

Le protocole à documenter est donc :

1. fixer un cas test simple, par exemple un sweep 1D en `alpha` à Mach fixé
2. injecter des points classiques seulement sur `c_i`
3. faire varier le nombre de points supervisés
4. mesurer :
   - l'erreur sur `c_i`
   - la qualité du mode reconstruit
   - la stabilité du résultat entre runs
5. retenir le plus petit nombre de points donnant une qualité jugée suffisante

### Message scientifique à retenir

Le message n'est pas :

- "le PINN ne marche pas sans solveur classique"

Le message est :

- "le PINN reconstruit bien le mode en physique pure, mais la valeur propre imaginaire reste mal identifiée sans ancrage spectral"

Dans ce cadre, un hybride minimal sur `c_i` est une décision méthodologiquement propre, parce qu'il cible exactement le verrou restant au lieu de superviser inutilement tout le problème.
