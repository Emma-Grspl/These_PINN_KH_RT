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
- préparation de variantes stabilisées et variantes inspirées du système de Riccati

### Ce que les expériences ont montré

- le verrou principal n'est plus `c_i`, mais le mode
- l'amplitude est mauvaise surtout à bas `alpha`, approximativement pour `alpha < 0.45`
- la phase reste mauvaise sur une plage encore plus large, approximativement jusqu'à `alpha < 0.6`
- le run `modefocus_lowalpha` améliore un peu la phase
- cette amélioration se paie par une dégradation partielle de l'amplitude
- une faible erreur scalaire ne suffit pas à conclure que le mode est bien reconstruit

### Difficultés rencontrées

- la physique pure ne verrouille pas suffisamment le mode dans la formulation actuelle
- la phase est la partie la plus fragile
- amplitude et phase ne progressent pas forcément ensemble
- le découpage amplitude/phase peut introduire des ambiguïtés d'optimisation
- la formulation premier ordre réel testée jusqu'ici a montré un comportement instable et des erreurs modales très élevées

### Position méthodologique actuelle

- figer le baseline `c_i` avec 8 points de supervision légère
- ne pas surpromettre la reconstruction modale tant que les courbes d'erreur de mode ne sont pas bonnes
- utiliser le classique comme vérité terrain explicite pour amplitude et phase

### Ce qu'il reste à faire ensuite

- finaliser l'évaluation des variantes stabilisées premier ordre réel et des variantes Riccati
- décider si la sortie réseau doit rester en amplitude/phase ou passer à une autre représentation réelle plus robuste
- mieux contrôler les conditions de bord et la normalisation modale
- introduire si nécessaire une supervision modale légère et ciblée, surtout à bas `alpha`
- une fois le cas `Mach` fixé satisfaisant, étendre le modèle à `(alpha, Mach)`

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

