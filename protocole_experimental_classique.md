# Protocole expérimental classique

Ce document résume l'état du solveur classique, séparément en subsonique et en supersonique.

## Subsonique

### Objectif

- obtenir une référence fiable pour `c_i`
- disposer d'une reconstruction modale utilisable comme vérité terrain pour le PINN

### Ce qui a déjà été fait

- mise en place d'un workflow hybride subsonique combinant un solveur principal et un solveur de contrôle plus robuste
- verrouillage d'une référence subsonique via [hybrid_subsonic_scan.py](/Users/emma.grospellier/Thèse/These_PINN_KH_RT/classical_solver/subsonic/hybrid_subsonic_scan.py)
- documentation du workflow de référence dans [classical_solver/subsonic/README.md](/Users/emma.grospellier/Thèse/These_PINN_KH_RT/classical_solver/subsonic/README.md)
- production d'artefacts de référence :
  - [subsonic_hybrid_growth_map.csv](/Users/emma.grospellier/Thèse/These_PINN_KH_RT/assets/classic_subsonic/data/subsonic_hybrid_growth_map.csv)
  - [subsonic_hybrid_vs_blumen.png](/Users/emma.grospellier/Thèse/These_PINN_KH_RT/assets/classic_subsonic/plots/subsonic_hybrid_vs_blumen.png)
  - [subsonic_hybrid_error_summary.json](/Users/emma.grospellier/Thèse/These_PINN_KH_RT/assets/classic_subsonic/data/subsonic_hybrid_error_summary.json)
- usage du classique subsonique comme référence pour comparer les modes PINN sur des sweeps en `alpha` à `Mach` fixé

### Pistes déjà explorées

- solveur principal de tir rapide : [shooting_subsonic.py](/Users/emma.grospellier/Thèse/These_PINN_KH_RT/classical_solver/subsonic/shooting_subsonic.py)
- solveur de contrôle robuste près de la neutralité : [mstab17_subsonic_solver.py](/Users/emma.grospellier/Thèse/These_PINN_KH_RT/classical_solver/subsonic/mstab17_subsonic_solver.py)
- comparaison systématique entre solveurs
- reconstruction de cartes et de profils modaux servant de référence au PINN

### Difficultés rencontrées

- la zone proche de la neutralité demande un traitement plus robuste que le tir standard
- la référence sur `c_i` est bien plus simple à verrouiller que la reconstruction modale complète
- pour les comparaisons PINN, il faut faire attention aux métriques amplitude/phase, sinon on peut conclure trop vite que le mode est bon

### Ce qu'on considère comme acquis

- la référence subsonique sur `c_i` est suffisamment solide pour servir de base de comparaison
- le workflow hybride est la référence classique subsonique actuelle

### Ce que les tests PINN ont clarifié sur la difficulté subsonique

Les derniers tests ne pointent pas vers une difficulté uniforme en `alpha`. Ils montrent au contraire trois régimes distincts :

| Régime | Observation | Lecture |
| --- | --- | --- |
| `alpha ≈ 0.20-0.80` | Le mode est globalement retrouvable avec la physique standard | Le verrou principal reste la qualité de reconstruction, pas l'existence de la bonne branche |
| `alpha ≈ 0.05` | Le pur physique dérive, mais un guidage classique léger répare fortement les métriques modales | Le problème vient surtout d'une mauvaise identifiabilité du mode à faible croissance |
| `alpha ≈ 0.85` | Le bon `c_i` devient atteignable, mais la bonne famille modale reste perdue | Le problème n'est plus l'eigenvaleur, mais la sélection de branche modale |

Résultats marquants déjà obtenus :

- `low-alpha targeted` à `alpha = 0.05` :
  - `target_p_rel ≈ 1.90e-01`
  - `band_p_rel ≈ 1.98e-01`
  - donc la branche basse n'est pas perdue, mais elle demande un guidage classique léger
- `high-alpha targeted`, `continuation` et `stepwise` à `alpha = 0.85` :
  - le bon `c_i` peut devenir presque correct
  - mais `p_rel` reste de l'ordre de `1.0`
  - donc le mode reste sur la mauvaise famille
- `high-alpha mode repair` avec `c_i` gelé :
  - pas d'amélioration substantielle au point cible
  - ce test confirme que le verrou haut-alpha n'est pas un simple réglage local de loss

### Pourquoi le pur physique ne suffit pas aux extrémités

- à bas `alpha`, la croissance est faible et la branche devient mal identifiable par les seules pertes physiques
- plusieurs profils peuvent satisfaire le résidu avec un coût proche tout en restant faux en enveloppe, en localisation ou en phase
- à haut `alpha`, la difficulté change de nature :
  - le scalaire `c_i` n'est plus le vrai verrou
  - plusieurs familles modales voisines peuvent porter un `c_i` proche
  - les pertes physiques et les diagnostics moyens sur une bande d'`alpha` n'imposent pas assez fortement la bonne structure
- un entraînement PINN sur bande subit aussi un compromis global :
  - il peut améliorer la moyenne de bande
  - tout en sacrifiant le point extrême `alpha = 0.85`

### Pourquoi le classique, et en particulier le shooting, tient mieux la branche

- le solveur classique traite un couple `(alpha, Mach)` à la fois
- il impose directement les conditions asymptotiques du problème
- il n'a pas à partager sa capacité entre plusieurs `alpha`
- il n'optimise pas une loss moyenne : il résout un problème spectral local
- en pratique, cela suffit à verrouiller proprement les quantités de référence là où le PINN peut encore confondre plusieurs familles modales

### Choix méthodologique retenu pour le PINN subsonique

Le protocole retenu n'est donc pas un schéma unique sur toute la bande en `alpha`. Il est explicitement découpé par régimes :

- `alpha ≈ 0.20-0.80` :
  - physique standard / hybride standard
- `alpha ≈ 0.05` :
  - guidage classique léger sur `c_i`, `q` et l'ancrage spatial
- `alpha ≈ 0.85` :
  - supervision classique explicite du mode

En une phrase :

- au bas-alpha, il faut guider l'eigenvaleur et la localisation
- au haut-alpha, il faut guider la famille modale elle-même

### Ce qu'il reste à faire

- formaliser proprement la référence de mode subsonique si l'objectif est de publier des comparaisons modales détaillées
- décider si la référence classique de mode doit être stockée sur une grille complète `(alpha, Mach)` ou seulement générée à la demande
- garder la cohérence entre les métriques de comparaison du classique et celles utilisées côté PINN
- fournir au prochain protocole PINN subsonique une base 2D `(alpha, Mach) -> c_i` assez propre pour reconstruire des isolignes

## Supersonique

### Objectif

- retrouver la bonne branche instable en supersonique
- produire une référence fiable en `c_r`, `c_i` et mode

### Ce qui a déjà été fait

- mise en place d'un solveur de tir supersonique et de plusieurs audits locaux
- mise en place d'un solveur GEP dense et d'outils d'exploration spectrale
- essais de sélection de branche :
  - audit shooting vs familles
  - sélecteur modal calibré
  - continuation de branche
  - beam search multi-branches
  - clustering de familles modales
- redigitalisation propre des isolignes Blumen supersoniques pour `c_r` et `c_i`
- centralisation des références Blumen dans [blumen_reference.py](/Users/emma.grospellier/Thèse/These_PINN_KH_RT/classical_solver/supersonic/blumen_reference.py)
- audit des familles contre Blumen avec [audit_supersonic_families_against_blumen.py](/Users/emma.grospellier/Thèse/These_PINN_KH_RT/scripts/audit_supersonic_families_against_blumen.py)
- sweep GEP guidé par Blumen avec [run_supersonic_gep_blumen_guided_sweep.py](/Users/emma.grospellier/Thèse/These_PINN_KH_RT/scripts/run_supersonic_gep_blumen_guided_sweep.py)
- ajout d'un diagnostic du spectre brut contre Blumen avec [diagnose_supersonic_raw_spectrum_vs_blumen.py](/Users/emma.grospellier/Thèse/These_PINN_KH_RT/scripts/diagnose_supersonic_raw_spectrum_vs_blumen.py)

### Résultats déjà clarifiés

- les figures supersoniques de Blumen représentent bien `c_r` et `c_i`, pas `omega_i`
- le shooting retrouve une branche instable auto-cohérente sur `alpha = 0.2` et `Mach = 1.2, 1.25, 1.275, 1.3`
- cette branche est cohérente asymptotiquement et reproduit correctement `c_i`
- les familles GEP sélectionnées jusque-là ne reproduisent pas la bonne croissance
- le sweep guidé par Blumen trouve un `c_r` souvent plausible, mais un `c_i` trop faible, de l'ordre de `0.006-0.008` au lieu de `0.047-0.062`
- le diagnostic du spectre brut contre Blumen montre `n_inside_blumen_box = 0` sur les cas cibles `alpha = 0.2`, `Mach = 1.2, 1.25, 1.275, 1.3`
- les meilleurs modes bruts en `c_r` restent bloqués vers `c_i ≈ 0.003-0.004`, donc bien en dessous du mode instable attendu
- la comparaison directe candidats GEP vs mode shooting confirme que l'écart principal n'est pas un mauvais tri local : les candidats GEP proches en `c_r` restent très loin du bon `c_i`
- l'audit d'eigenconditions montre que l'écart résiduel à Blumen est presque entièrement porté par `c_r`, pas par `c_i`
- l'audit local de la référence Blumen et le bootstrap d'incertitude montrent que `c_r` est localement sous-contraint autour de `alpha = 0.2`, `Mach = 1.25-1.30`

### Pistes déjà explorées

- choisir la branche la plus instable
- choisir la branche la plus proche du shooting
- choisir la branche la plus cohérente modalement avec une référence locale
- imposer des scores sur continuité, overlap modal, distance spectrale et seuils sur `c_r`
- guider la sélection avec les valeurs Blumen redigitalisées

### Ce que ces essais ont montré

- le problème principal n'est plus seulement le sélecteur de branche
- le GEP brut ne fait pas apparaître la branche fortement instable recherchée aux points audités
- en pratique, on observe une famille GEP à `c_r` parfois plausible mais à `c_i` trop faible, donc une branche quasi neutre qui peut mimer la bonne vitesse de phase sans porter la bonne croissance
- il faut donc distinguer deux questions :
  - la branche Blumen existe-t-elle dans le spectre brut
  - si oui, pourquoi n'est-elle pas sélectionnée

### Pourquoi on bascule de GEP vers shooting

- on ne s'acharne pas sur le GEP à ce stade parce que le dernier verrou n'est plus un problème de ranking, mais un problème d'existence de la bonne branche instable dans le spectre exploité
- tant que le spectre brut ne montre pas de mode proche de Blumen simultanément en `c_r` et en `c_i`, raffiner le sélecteur GEP ne peut pas régler le fond du problème
- le shooting est donc utilisé comme solveur de référence pour verrouiller une branche physiquement cohérente le long d'une ligne de Mach
- une fois cette branche shooting stabilisée, elle sert à répondre proprement à la question importante :
  - le GEP rate-t-il seulement la sélection
  - ou bien la formulation GEP actuelle manque réellement la branche instable
- en d'autres termes, le shooting n'est pas un contournement définitif du GEP ; c'est l'outil le plus direct pour établir d'abord une branche de référence crédible, puis auditer le GEP contre elle

### Difficultés rencontrées

- coexistence de plusieurs familles modales en supersonique
- confusion initiale sur la calibration de l'axe Mach des données Blumen digitalisées
- divergence possible entre le shooting et le GEP selon le Mach
- risque de confondre une branche quasi neutre avec la bonne branche physique si on regarde surtout `c_r`

### Ce qu'il reste à faire

- le shooting est maintenant la référence pratique figée
- le GEP est mis de côté comme outil de diagnostic spectral, pas comme solveur de vérité
- les prochains travaux classiques supersoniques ne sont plus prioritaires avant l'ouverture du chantier PINN supersonique
- si on revient au GEP plus tard, ce sera pour une reformulation des bords, du mapping ou de la résolution, pas pour raffiner encore le ranking courant

### Position méthodologique retenue

- Blumen sert de référence externe
- le solveur classique ne doit pas dépendre de Blumen pour produire ses solutions
- Blumen sert à valider la bonne branche, pas à la fabriquer

### Décision scientifique retenue

En régime supersonique, le solveur classique de référence retenu est désormais le `shooting`, et non le GEP. Cette décision repose sur le fait que le shooting retrouve une branche instable auto-cohérente, satisfait correctement les conditions asymptotiques de la formulation, et reproduit de manière robuste la croissance `c_i`, qui est la quantité la plus fiable dans la comparaison à Blumen. En revanche, le `c_r` issu des courbes digitalisées de Blumen autour de `alpha = 0.2` et `Mach = 1.25-1.30` apparaît localement trop peu contraint pour servir de vérité ponctuelle stricte. Le GEP standard n'étant pas capable de retrouver la bonne croissance `c_i`, il n'est pas retenu comme solveur classique de validation supersonique.

| Point | Statut | Lecture |
| --- | --- | --- |
| Existence d'une branche instable supersonique | Validé | Le shooting converge de façon stable sur une branche cohérente |
| Conditions asymptotiques et raccord | Validé | Les audits montrent une bonne cohérence interne du shooting |
| Reproduction de `c_i` de Blumen | Validé | Accord globalement bon, donc benchmark principal crédible |
| Reproduction point par point de `c_r` de Blumen | Non validé strictement | La référence digitalisée en `c_r` est localement trop incertaine |
| GEP comme solveur classique supersonique de référence | Non validé | Il rate la croissance `c_i`, donc il ne peut pas servir de vérité |
| Usage de `c_r` comme cible quantitative forte | Non retenu | À utiliser seulement qualitativement ou avec barre d'incertitude |
| Structure modale supersonique | À consolider | À comparer explicitement à la forme attendue du mode |

### Prochaine étape retenue sur le shooting

- prendre `c_i` comme cible principale de calibration sur les courbes Blumen
- comparer explicitement la structure du mode obtenu par shooting à la forme physique attendue
- utiliser des critères de continuité modale et d'overlap central pour suivre la branche en `Mach`
- ne plus forcer un alignement point par point sur `c_r` tant que l'incertitude locale de la référence Blumen domine l'écart
- garder le GEP comme audit secondaire, pas comme solveur de vérité
