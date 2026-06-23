# Référence Classique

Ce dossier contient les points d'entrée humains pour lancer ou reconstruire les références classiques.

Les modules réels restent volontairement à leur emplacement actuel :

- `classical_solver/` contient les solveurs et briques numériques centrales.
- `scripts/` contient encore plusieurs helpers partagés importés par d'autres scripts.

Cette organisation évite de casser les imports existants pendant le nettoyage du dépôt. Les fichiers ici sont donc des wrappers minces : ils redirigent vers les scripts actuels et préservent les arguments CLI.

## Subsonique

La référence subsonique est considérée comme figée. Les wrappers subsoniques servent surtout à retrouver facilement les points d'entrée de reconstruction ou de scan classique.

## Supersonique

La référence supersonique repose principalement sur le shooting pointwise, avec des critères de robustesse de boîte et une validation spectrale/modale éparse.

Le GEP reste un outil de diagnostic utile pour inspecter des familles spectrales, comparer des candidats ou comprendre les branches numériques. Il n'est pas la référence principale pour le supersonique.

