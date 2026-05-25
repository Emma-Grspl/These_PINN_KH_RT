# Protocole PINN subsonique

Ce document fixe la stratégie subsonique à court terme et documente précisément la supervision actuellement utilisée dans le PINN.

## 1. Objectif scientifique

L'objectif n'est pas de couvrir immédiatement tout le domaine subsonique.  
L'objectif utile est :

- obtenir un PINN capable de reconstruire un bon `c_i`
- et un bon mode `(p, rho, \hat{u}, \hat{v})`
- pour `0.05 <= alpha <= 0.8`
- et, à terme, pour `0 <= Mach <= 1`

sur un domaine où la physique est encore franchement instable et où la reconstruction modale reste numériquement raisonnable.

## 2. Domaine de travail

On distingue maintenant trois sous-domaines.

### 2.1 Domaine coeur

`D_core = {0.05 <= alpha <= 0.8, 0 <= Mach <= 1}`

C'est le domaine sur lequel on veut d'abord :

- un `c_i` fiable
- des modes fiables
- des heatmaps d'erreur lisibles en fonction de `(alpha, Mach)`

### 2.2 Bande très bas alpha

`D_low = {alpha < 0.05}`

Cette zone est reportée à un raffinement ultérieur parce que :

- `\hat{u}` et `\hat{v}` y sont mal conditionnés
- la moindre erreur sur `p_y / p` y est fortement amplifiée

### 2.3 Bande haute alpha

`D_high = {alpha > 0.8}`

Cette zone est aussi reportée à un raffinement ultérieur parce qu'on se rapproche de la neutralité, ce qui dégrade surtout la reconstruction de `\hat{u}`.

## 3. Feuille de route

L'ordre retenu est le suivant.

### 3.1 Étape A

Valider d'abord une reconstruction **1D** à `Mach = 0.5` sur `0.05 <= alpha <= 0.8` :

- `c_i` correct
- `p` et `rho` corrects
- `\hat{v}` correct ou quasi correct
- `\hat{u}` sans explosion

Cette étape 1D doit être bonne à la fois :

- sur la reconstruction spectrale `c_i`
- sur la reconstruction modale

avant toute extension au cas 2D.

### 3.2 Étape B

Une fois la reconstruction 1D validée, étendre la même logique à quelques tranches en Mach :

- `Mach = 0.2`
- `Mach = 0.5`
- `Mach = 0.8`

### 3.3 Étape C

Passer ensuite à un vrai PINN **2D** `(alpha, Mach)` sur `D_core`, avec :

- heatmap d'erreur `c_i(alpha, Mach)`
- heatmaps d'erreur modale `L2_mean(alpha, Mach)` pour `p`, `rho`, `\hat{u}`, `\hat{v}`

### 3.4 Étape D

Une fois le protocole supervisé validé en 1D puis en 2D, relancer **exactement le même protocole sans supervision classique** pour mesurer ce qu'on perd sans guidage classique.

Le but de cette étape est explicite :

- montrer que la supervision classique est nécessaire
- quantifier ce qu'elle apporte sur `c_i`
- quantifier ce qu'elle apporte sur la reconstruction modale

Autrement dit, il faut prévoir une vraie étude d'ablation :

- protocole avec supervision classique sparse
- même protocole sans supervision classique

### 3.5 Étape E

Raffiner ensuite seulement :

- `alpha < 0.05`
- `alpha > 0.8`

## 4. Supervision de `c_i`

Il faut distinguer le **trainer générique** et le **run low-alpha Riccati actuel**.

### 4.1 Principe générique

Dans le trainer 1D à Mach fixé, `alpha` est une entrée et `c_i` est une sortie :

`alpha -> c_i^{pred}(alpha)`

Une référence classique est préconstruite sur une grille en `alpha` via la cache classique subsonique.  
À chaque epoch, on tire un batch `alpha_supervision` puis on calcule :

- `c_i^{target} = c_i^{classic}(alpha_supervision)`
- `c_i^{pred} = model.get_ci(alpha_supervision)`
- `loss_ci = mean((c_i^{pred} - c_i^{target})^2)`

La loss effectivement injectée dans l'entraînement est :

`w_ci_supervision * loss_ci`

Le terme correspondant dans le code est :

- `loss_ci = torch.mean((ci_pred - ci_target).pow(2))`

### 4.2 Comment les points `alpha` sont choisis

Les points ne sont pas fixes. Ils sont tirés par `sample_alpha_adaptive_batch(...)`.

Cette routine mélange trois familles de points :

- points uniformes sur `[alpha_min, alpha_max]`
- points centrés autour de `focus_alphas`
- points optionnels autour d'une zone de neutralité

Les paramètres utiles sont :

- `n_alpha_supervision`
- `focus_fraction`
- `focus_half_width`
- `neutral_fraction`
- `neutral_alpha`
- `neutral_half_width`

La logique est donc :

- une fraction du batch est uniforme
- une fraction du batch est concentrée autour des alphas jugés difficiles
- éventuellement une petite fraction est forcée près de la neutralité

### 4.3 Comment les `focus_alphas` sont mis à jour

À chaque audit :

- on évalue l'erreur `c_i`
- on évalue l'erreur modale
- on collecte les alphas en échec

Plus précisément :

- les alphas où `ci_abs_err > error_threshold`
- et les alphas où `mode_rel_err > mode_error_threshold`

forment la liste candidate `failing_alphas`.

Si cette liste est trop longue, on garde les `max_focus_points` plus sévères avec un score :

`audit_ci_weight * ci_abs_err + audit_mode_weight * mode_rel_err`

Ces `focus_alphas` servent ensuite au sur-échantillonnage de l'epoch suivante.

### 4.4 Important : dans le run low-alpha actuel, `c_i` n'est pas supervisé

Dans le stage `riccati_lowalpha_gamma_repair`, on ne réentraîne pas la branche `c_i`.

Les choix effectifs sont :

- `freeze_ci = True`
- `enable_classic_ci_supervision = False`
- `w_ci_supervision = 0.0`

Donc :

- `c_i` est figé au warm start
- il n'y a pas de `loss_ci` active
- le stage ne répare que les **modes**

## 5. Supervision des modes

Le run subsonique actuel utilise la représentation **Riccati**.  
Le réseau modal ne sort pas directement `p, rho, \hat{u}, \hat{v}`.  
Il sort :

- `kappa(xi, alpha)`
- `q(xi, alpha)`

avec :

- `gamma = kappa + i q = p_y / p`

### 5.1 Loss PDE principale

La loss intérieure impose l'équation de Riccati :

- résidu réel
- résidu imaginaire

Le terme injecté est :

`loss_pde = mean(res_r^2 + res_i^2)`

pondéré par :

- `w_pde`

Dans la pratique actuelle, le warm start impose déjà les autres normalisations, donc le poids principal reste `w_pde = 1`.

### 5.2 Conditions asymptotiques aux bords

Le modèle est contraint aux deux bords via :

- `loss_bc_kappa`
- `loss_bc_q`

Ces termes imposent la bonne décroissance asymptotique de `gamma` à gauche et à droite.

La loss injectée est :

- `w_bc_kappa * loss_bc_kappa`
- `w_bc_q * loss_bc_q`

### 5.3 Boundary bands

On ajoute une contrainte plus robuste dans une bande proche des bords :

- `loss_riccati_boundary_band_kappa`
- `loss_riccati_boundary_band_q`

Les paramètres actuels sont :

- `riccati_boundary_band_points = 64`
- `riccati_boundary_band_start = 0.94`
- `riccati_boundary_band_end = 0.995`

L'idée est de ne pas imposer seulement l'extrémité stricte, mais toute une queue asymptotique.

### 5.4 Matching shooting-like

Le terme le plus proche du shooting est :

- `loss_riccati_shooting_match`

Il compare le prolongement depuis la gauche et depuis la droite, puis pénalise le mismatch au centre.

C'est le terme le plus directement lié au raccord du mode.

### 5.5 Ancre Riccati sparse

On utilise aussi une supervision sparse sur la pression reconstruite depuis `gamma` :

- `loss_riccati_anchor`

Pour chaque alpha d'ancrage :

1. on reconstruit `p` à partir du champ Riccati
2. on interpole la pression classique de référence
3. on calcule une MSE sur les parties réelle et imaginaire

La loss correspondante est une moyenne des écarts quadratiques sur un petit intervalle centré.

### 5.6 Supervision sparse sur `q`

On utilise aussi :

- `loss_q_supervision`

La référence classique est :

- `q_ref = Im(p_y / p)`

obtenue par dérivation numérique du mode classique.

La procédure est :

1. on choisit quelques `alpha`
2. on prend une grille `xi_template`
3. on la convertit en `y`
4. on interpole `q_ref(y)` sur la zone de recouvrement
5. on masque les points où l'enveloppe de `p_ref` est trop faible
6. on calcule une MSE sur `q`

Donc la supervision n'est pas faite :

- partout en `y`
- ni partout en `alpha`

Elle est :

- sparse en `alpha`
- sparse en `y`
- limitée à la zone où le mode a une amplitude suffisante

### 5.7 Supervision sparse sur `gamma`

Le stage le plus récent ajoute :

- `loss_riccati_gamma_supervision`

Ici, on supervise :

- `kappa = Re(gamma)`
- `q = Im(gamma)`

La loss locale utilisée est :

`0.5 * (MSE(kappa_pred, kappa_ref) + MSE(q_pred, q_ref))`

comme pour `q`, uniquement :

- sur quelques `alpha`
- sur une grille `xi` fine
- avec un masque d'amplitude
- et une surpondération à bas `alpha`

### 5.8 Surpondération bas alpha

Les losses `q` et `gamma` sont pondérées par :

- `low_alpha_weight` si `alpha <= low_alpha_threshold`
- `1` sinon

Dans le run actuel :

- `low_alpha_threshold = 0.5`
- `low_alpha_weight = 8`

Donc la supervision n'est **pas uniforme** sur tout le domaine.

## 6. Quels points sont effectivement supervisés dans le run actuel

Le run `kh_subsonic_fixed_mach_M05_riccati_lowalpha_gamma_repair_w3` utilise :

- `anchor_alphas = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]`
- `q_supervision_alphas = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]`
- `gamma_supervision_alphas = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]`

Ces listes sont **explicites**.  
Elles ne sont pas tirées aléatoirement dans ce run.

En revanche, les points `alpha_interior`, `alpha_boundary`, `alpha_ref`, `alpha_norm` utilisés pour :

- la PDE
- les conditions aux bords
- le matching
- les audits internes

sont toujours tirés par `sample_alpha_adaptive_batch(...)`.

## 7. Poids effectifs du run low-alpha gamma repair

Les poids actifs du run actuel sont :

- `w_q_supervision = 5`
- `w_riccati_gamma_supervision = 3`
- `w_riccati_anchor = 2`
- `w_riccati_boundary_band_kappa = 5`
- `w_riccati_boundary_band_q = 10`
- `w_riccati_shooting_match = 12`

Poids désactivés côté spectral :

- `w_ci_supervision = 0`

Paramètres de fréquence :

- `riccati_anchor_every = 20`
- `q_supervision_every = 10`
- `riccati_gamma_every = 10`

Tailles de grilles utilisées :

- `n_interior = 512`
- `n_boundary = 64`
- `n_anchor_alpha = 24`
- `n_norm_interior = 256`
- `n_alpha_supervision = 32` mais inactif ici puisque `c_i` est gelé
- `riccati_anchor_n_xi = 97`
- `q_supervision_n_xi = 129`
- `riccati_gamma_n_xi = 129`

Paramètres d'échantillonnage en `alpha` :

- `focus_fraction = 0.85`
- `focus_half_width = 0.04`
- `max_focus_points = 12`

## 8. Est-ce supervisé partout pareil ?

Non.

Il faut distinguer quatre niveaux.

### 8.1 Pas la même supervision selon la branche

- branche `c_i` : gelée dans le run actuel
- branche modale : fortement contrainte

### 8.2 Pas la même supervision selon `alpha`

- bas `alpha` : surpondéré par `low_alpha_weight = 8`
- haut `alpha` : poids nominal

### 8.3 Pas la même supervision selon `y`

Les losses `q` et `gamma` utilisent un masque d'amplitude :

- si l'enveloppe du mode classique est trop faible, le point n'est pas utilisé

Donc on ne supervise pas les queues trop faibles comme si elles valaient autant que le coeur du mode.

### 8.4 Pas la même supervision selon l'époque

Certaines losses ne sont pas actives à chaque epoch :

- ancre Riccati toutes les `20` epochs
- `q` toutes les `10` epochs
- `gamma` toutes les `10` epochs

Donc le guidage classique reste sparse dans le temps aussi.

## 9. Stratégie retenue à partir de maintenant

La stratégie subsonique retenue est maintenant :

### 9.1 D'abord réussir la reconstruction 1D sur `D_core`

On demande d'abord au PINN d'être bon en 1D sur :

- `0.05 <= alpha <= 0.8`
- `Mach = 0.5`

avec supervision classique minimale.

Le critère n'est pas seulement un bon `c_i`.

Il faut aussi :

- un bon mode
- une bonne phase
- une bonne structure spatiale

### 9.2 Étendre ensuite vers le 2D

Une fois la version 1D bonne, on étend la même logique à des Mach multiples puis au 2D sur :

- `0.05 <= alpha <= 0.8`
- `0 <= Mach <= 1`

### 9.3 Produire des heatmaps 2D

Les sorties cibles de la prochaine étape 2D sont :

- heatmap erreur `c_i(alpha, Mach)`
- heatmap `L2_mean(alpha, Mach)` pour `p`
- heatmap `L2_mean(alpha, Mach)` pour `rho`
- heatmap `L2_mean(alpha, Mach)` pour `\hat{u}`
- heatmap `L2_mean(alpha, Mach)` pour `\hat{v}`

### 9.4 Faire ensuite l'ablation sans supervision

Une fois le protocole supervisé validé :

- en 1D
- puis en 2D

on relance le même protocole sans supervision classique pour démontrer explicitement que cette supervision est nécessaire.

Cette comparaison doit être faite à protocole identique autant que possible :

- même architecture
- même maillage expérimental
- mêmes métriques
- mêmes heatmaps

La différence visée doit porter uniquement sur le retrait du guidage classique.

### 9.5 Raffiner ensuite les bandes extrêmes

Une fois `D_core` validé, on traite à part :

- `alpha < 0.05`
- `alpha > 0.8`

Le raffinement ne doit pas contaminer le benchmark principal.

## 10. Conclusion méthodologique

Le point clé est le suivant :

- on ne veut pas superviser plus
- on veut superviser mieux

Donc :

- pas de supervision full-mode dense partout
- pas de régression brute sur `\hat{u}` et `\hat{v}` sur tout le domaine
- oui à une supervision sparse sur `q` et `gamma`
- oui à une hiérarchie par domaines et par régimes

La logique du protocole subsonique est maintenant :

1. verrouiller la reconstruction 1D sur `D_core`
2. étendre cette reconstruction au 2D `alpha-Mach`
3. relancer le même protocole sans supervision classique pour l'ablation
4. seulement ensuite raffiner les bandes extrêmes
