Meilleur cas figé pour le PINN subsonique à `Mach` fixé avec sweep en `alpha`.

Référence retenue :

- protocole `classic_two_stage_repair`
- meilleur checkpoint : `stage_b_pressure_overlap/model_best.pt`

Contenu :

- `classic_two_stage_repair_summary.csv` : résumé global du run retenu
- `stage_b_pressure_overlap/config.csv` : configuration du meilleur stage
- `stage_b_pressure_overlap/history.csv` : historique d'entraînement du meilleur stage
- `stage_b_pressure_overlap/model_best.pt` : checkpoint retenu
- `stage_b_pressure_overlap/stage_summary.csv` : métriques finales du meilleur stage
- `KH_PINN_M05_highalpha_two_stage_862263.out/.err` : log Jean Zay du run retenu

Le run `classic_two_stage_repair_refine` n'est pas retenu comme référence principale. Il a servi uniquement de test de sensibilité.
