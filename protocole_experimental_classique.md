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
  - [subsonic_hybrid_growth_map.csv](/Users/emma.grospellier/Thèse/These_PINN_KH_RT/assets/blumen_shooting_hybrid/subsonic_hybrid_growth_map.csv)
  - [subsonic_hybrid_vs_blumen.png](/Users/emma.grospellier/Thèse/These_PINN_KH_RT/assets/blumen_shooting_hybrid/subsonic_hybrid_vs_blumen.png)
  - [subsonic_hybrid_error_summary.json](/Users/emma.grospellier/Thèse/These_PINN_KH_RT/assets/blumen_shooting_hybrid/subsonic_hybrid_error_summary.json)
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

### Ce qu'il reste à faire

- formaliser proprement la référence de mode subsonique si l'objectif est de publier des comparaisons modales détaillées
- décider si la référence classique de mode doit être stockée sur une grille complète `(alpha, Mach)` ou seulement générée à la demande
- garder la cohérence entre les métriques de comparaison du classique et celles utilisées côté PINN

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
- le shooting par défaut semble cohérent avec Blumen pour `alpha = 0.2` et `Mach = 1.2, 1.25, 1.275`
- à `Mach = 1.3`, le shooting décroche vers une autre branche
- les familles GEP sélectionnées jusque-là ne reproduisent pas la bonne croissance
- le sweep guidé par Blumen trouve un `c_r` souvent plausible, mais un `c_i` trop faible, de l'ordre de `0.006-0.008` au lieu de `0.047-0.062`
- le diagnostic du spectre brut contre Blumen montre `n_inside_blumen_box = 0` sur les cas cibles `alpha = 0.2`, `Mach = 1.2, 1.25, 1.275, 1.3`
- les meilleurs modes bruts en `c_r` restent bloqués vers `c_i ≈ 0.003-0.004`, donc bien en dessous du mode instable attendu
- la comparaison directe candidats GEP vs mode shooting confirme que l'écart principal n'est pas un mauvais tri local : les candidats GEP proches en `c_r` restent très loin du bon `c_i`

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

- consolider la continuation shooting sur une ligne de Mach pour verrouiller simultanément `c_r` et `c_i`
- utiliser cette branche shooting comme référence d'audit contre le GEP
- si le GEP reste loin en `c_i`, revoir la formulation GEP, la résolution ou les paramètres numériques
- si une branche GEP pertinente réapparaît après cela, seulement alors revenir à un protocole de sélection/ranking
- ensuite verrouiller un protocole valable sur une grille `(alpha, Mach)`
- une fois la bonne branche trouvée, reconstruire des modes et refaire les isolignes sans utiliser Blumen comme seed, seulement comme audit

### Position méthodologique retenue

- Blumen sert de référence externe
- le solveur classique ne doit pas dépendre de Blumen pour produire ses solutions
- Blumen sert à valider la bonne branche, pas à la fabriquer
