# Reference PINN subsonique courante a `M = 0.5`

Cette reference fige le meilleur candidat 1D actuel pour le cas subsonique a `Mach = 0.5`.

Source retenue :

- run source :
  [kh_subsonic_fixed_mach_M05_riccati_mode_repair_edges_v2](/Users/emma.grospellier/Thèse/These_PINN_KH_RT/assets/pinn_subsonic/experiment_mode_repair_edges_v2_2026-06-02/model_saved/kh_subsonic_fixed_mach_M05_riccati_mode_repair_edges_v2)
- checkpoint retenu :
  [model_best.pt](/Users/emma.grospellier/Thèse/These_PINN_KH_RT/assets/pinn_subsonic/mach_fixed/frozen_M05_riccati_reference_current/model_best.pt)

Contenu fige :

- [model_best.pt](/Users/emma.grospellier/Thèse/These_PINN_KH_RT/assets/pinn_subsonic/mach_fixed/frozen_M05_riccati_reference_current/model_best.pt)
- [config.csv](/Users/emma.grospellier/Thèse/These_PINN_KH_RT/assets/pinn_subsonic/mach_fixed/frozen_M05_riccati_reference_current/config.csv)
- [history.csv](/Users/emma.grospellier/Thèse/These_PINN_KH_RT/assets/pinn_subsonic/mach_fixed/frozen_M05_riccati_reference_current/history.csv)
- [warmstart_source.csv](/Users/emma.grospellier/Thèse/These_PINN_KH_RT/assets/pinn_subsonic/mach_fixed/frozen_M05_riccati_reference_current/warmstart_source.csv)
- [mode_repair_edges_summary.csv](/Users/emma.grospellier/Thèse/These_PINN_KH_RT/assets/pinn_subsonic/mach_fixed/frozen_M05_riccati_reference_current/mode_repair_edges_summary.csv)
- [mode_repair_edges_improvement.csv](/Users/emma.grospellier/Thèse/These_PINN_KH_RT/assets/pinn_subsonic/mach_fixed/frozen_M05_riccati_reference_current/mode_repair_edges_improvement.csv)
- les sorties `posttrain_eval` de `c_i` dans [ci](/Users/emma.grospellier/Thèse/These_PINN_KH_RT/assets/pinn_subsonic/mach_fixed/frozen_M05_riccati_reference_current/ci)
- les sorties `posttrain_eval` modales dans [modes](/Users/emma.grospellier/Thèse/These_PINN_KH_RT/assets/pinn_subsonic/mach_fixed/frozen_M05_riccati_reference_current/modes)

Motif de promotion :

- `c_i` conserve la qualite du warmstart :
  `ci_mae = 4.99e-4`
- le mode s'ameliore globalement sur `alpha in [0.2, 0.8]`
- gains observes sur la moyenne globale :
  - `p_rel_mean : 0.01207 -> 0.01049`
  - `u_rel_mean : 0.03010 -> 0.02700`
  - `v_rel_mean : 0.03057 -> 0.02761`
  - `phase_rmse_mean : 0.02652 -> 0.02342`

Reserve restante :

- la phase reste legerement moins bonne au bord haut, autour de `alpha >= 0.7`
- cette reference est donc retenue comme base courante, pas comme point final definitif

Usage recommande :

- base de comparaison pour les prochains Mach subsoniques
- warmstart prioritaire pour les evaluations et promotions ulterieures a `M = 0.5`
