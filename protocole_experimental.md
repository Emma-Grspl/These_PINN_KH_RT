# Protocole expérimental global

Ce document fixe l'ordre logique du projet. L'idée est de verrouiller une référence classique avant de demander au PINN de l'imiter ou de la reproduire par la physique.

## Étape 1. Référence classique subsonique

Objectif :

- disposer d'une référence fiable pour `c_i`
- disposer d'une reconstruction modale de référence exploitable pour comparer le PINN

État actuel :

- la référence `c_i` subsonique est déjà bien avancée via le workflow hybride
- la reconstruction de mode existe sur des cas ciblés et sert déjà de vérité terrain pour les comparaisons PINN

Livrables attendus :

- carte ou base de données subsonique de référence
- procédure stable de reconstruction de mode
- critères d'erreur pour amplitude et phase

## Étape 2. Référence classique supersonique

Objectif :

- disposer d'une référence fiable pour `c_i`
- disposer d'une référence utilisable pour `c_r`, avec incertitude explicite si nécessaire
- disposer d'une reconstruction modale cohérente sur la bonne branche

État actuel :

- les données Blumen `c_r` et `c_i` ont été redigitalisées proprement
- le shooting est retenu comme solveur classique de référence en supersonique
- l'accord en `c_i` est robuste et sert de benchmark principal
- le `c_r` digitalisé de Blumen est localement trop peu contraint pour servir de vérité ponctuelle stricte dans la zone `alpha = 0.2`, `Mach = 1.25-1.30`
- le GEP n'est pas retenu comme solveur de référence car il ne reproduit pas correctement `c_i`

Livrables attendus :

- base de référence `(alpha, Mach) -> (c_r, c_i, mode)` issue du shooting
- protocole de suivi de branche validé principalement sur `c_i` et sur la structure modale
- quantification explicite de l'incertitude locale sur `c_r` côté Blumen

## Étape 3. PINN subsonique sur `(alpha, Mach)` pour apprendre `c_i + mode`

Objectif :

- apprendre un opérateur ou un modèle conditionné par `(alpha, Mach)`
- sortir `c_i`
- sortir le mode associé avec une reconstruction amplitude/phase correcte

État actuel :

- un baseline fixe en Mach est disponible à `M = 0.5`
- la supervision légère de `c_i` est justifiée et calibrée
- la reconstruction modale reste le verrou principal

Livrables attendus :

- baseline PINN subsonique fiable sur une ligne `Mach` fixée
- extension en `Mach`
- audit d'erreur de mode en fonction de `alpha` et `Mach`

## Étape 4. PINN supersonique sur `(alpha, Mach)` pour apprendre `c_i + c_r + mode`

Objectif :

- apprendre simultanément `c_r`, `c_i` et le mode
- respecter la bonne branche physique
- comparer systématiquement au classique validé

Précondition :

- ne pas lancer sérieusement cette étape tant que la référence classique supersonique n'est pas verrouillée

Livrables attendus :

- formulation PINN supersonique stable
- protocole de branche explicite
- comparaison au classique sur `c_r`, `c_i` et mode

## Ordre de travail retenu

1. verrouiller le classique subsonique si nécessaire sur les métriques de mode
2. verrouiller le classique supersonique sur la bonne branche
3. consolider le PINN subsonique
4. seulement ensuite attaquer le PINN supersonique

## Critère de passage entre étapes

On ne passe pas à l'étape suivante parce qu'un script tourne, mais parce qu'on a :

- une référence calculable de manière répétable
- une comparaison quantitative à Blumen ou au classique
- une lecture claire des échecs restants
