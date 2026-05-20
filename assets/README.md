Ce dossier contient uniquement les sorties de référence encore actives.

Organisation actuelle :

- `classic_subsonic/`
  Référence classique subsonique retenue.
- `classic_supersonic/`
  Référence classique supersonique, audits de branche et comparaison à Blumen.
- `pinn_subsonic/`
  Sorties de référence du PINN subsonique retenues pour `c_i` et le mode.
- `pinn_supersonic/`
  Réservé pour la suite. Pas encore de référence figée.

Sous-structures utilisées :

- `data/` : CSV et résumés
- `plots/` : figures de synthèse
- `modes/` : reconstructions modales
- `diagnostics/` : audits complémentaires utiles pour justifier la sélection
- `blumen_reference/` : données Blumen redigitalisées
- `shooting/` : sorties de référence par tir, validations visuelles et expériences ciblées
- `gep/` : diagnostics GEP conservés pour comparaison spectrale
- `frozen_fixed_mach_alpha_sweep_best/` : meilleur run PINN subsonique figé pour `Mach` fixé et sweep en `alpha`

Statut du workspace actif :

- l'ancien `assets/blumen_gep/` n'existe plus
- les sorties de runs PINN obsolètes dans `model_saved/` ont été supprimées après gel du meilleur cas
- `pinn_supersonic/` reste volontairement vide de résultats, en attente du protocole supersonique

Les deux seules sources supersoniques Blumen conservées dans le repo de travail sont :

- `KH_RT_Blumen/supersonic/cr_datasets.csv`
- `KH_RT_Blumen/supersonic/ci_datasets.csv`
