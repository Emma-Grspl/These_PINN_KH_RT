Ce dossier contient uniquement les sorties de référence encore actives.

Organisation actuelle :

- `classique_subsonique/`
  Référence classique subsonique retenue.
- `classique_supersonique/`
  Référence classique supersonique, audits de branche et comparaison à Blumen.
- `pinn_subsonique/`
  Sorties de référence du PINN subsonique retenues pour `c_i` et le mode.
- `pinn_supersonique/`
  Réservé pour la suite. Pas encore de référence figée.

Sous-structures utilisées :

- `data/` : CSV et résumés
- `plots/` : figures de synthèse
- `modes/` : reconstructions modales
- `diagnostics/` : audits complémentaires utiles pour justifier la sélection
- `blumen_reference/` : données Blumen redigitalisées

Tout ce qui était exploratoire ou redondant a été déplacé vers :

- `archive/repo_cleanup_2026-04-24/`

En particulier :

- les anciens `assets/blumen/` et `assets/blumen_gep/`
- l'ancien `plot_presentation/`
- les sorties d'entraînement dans `model_saved/`

Les deux seules sources supersoniques Blumen conservées dans le repo de travail sont :

- `KH_RT_Blumen/supersonic/cr_datasets.csv`
- `KH_RT_Blumen/supersonic/ci_datasets.csv`
