# Supersonic Multicandidate Box Selector Report

## Files Created

- `scripts/audit_supersonic_shooting_blumen_locked_multicandidate.py`
- `launch/jz_submit_supersonic_shooting_blumen_locked_multicandidate.slurm`
- `repo_supersonic_multicandidate_box_selector_report.md`

## Files Modified

- aucun fichier existant de reference valide n'a ete modifie
- aucun script existant n'a ete deplace ou supprime

## Candidate Generation Logic

Le nouveau script fait un audit **par point `(alpha, Mach)`** avec plusieurs familles de seeds plausibles :

1. `existing`
   - ancres locales derivees de `supersonic_validated_modal_points.csv`
   - candidats recuperes depuis les CSV existants sous `assets/classic_supersonic/shooting/` quand leur format est reconnu

2. `branch_guided`
   - interpolation entre references spectrales/modales voisines quand le Mach cible est brackettable
   - reuse des seeds guidees deja presentes dans `audit_supersonic_shooting_point_batch_branch_guided.py`

3. `blumen_perturb`
   - cible Blumen directe si `c_i` est interpolable
   - perturbations raisonnables autour de cette cible en `c_r` et `c_i`
   - fallback sur la cible guidee si `c_r` Blumen n'est pas disponible mais qu'une branche guidee existe

Mode `candidate-source` :

- `existing` : seulement les seeds existantes
- `branch_guided` : seulement les seeds guidees
- `blumen_perturb` : seulement les perturbations Blumen
- `all` : union deduplicatee de toutes les familles
- `auto` : interleaving ordonne `existing -> branch_guided -> blumen_perturb`, ce qui force une vraie diversite de sources quand elles existent

## Acceptance Criteria

Un candidat est `strict_accepted` seulement si toutes les contraintes suivantes passent :

- `stage1_mismatch <= strict_stage1`
- `stage2_mismatch <= strict_stage2`
- `ci_abs_err <= strict_ci_abs`
- `ci_rel_err <= strict_ci_rel`
- `cr_abs_err <= strict_cr_abs` si une reference `c_r` existe
- robustesse de boite valide si `--box-required`
- `peak_shift <= strict_peak_shift` si la metrique est disponible

Raisons de rejet standardisees :

- `exception`
- `solver_failure`
- `missing_reference`
- `stage1`
- `stage2`
- `ci_abs`
- `ci_rel`
- `cr_abs`
- `box`
- `peak_shift`
- `no_candidate`

Le script remplit :

- `reject_reason_primary`
- `reject_reasons_all`

## Progressive Outputs

Le script ecrit dans :

- `assets/classic_supersonic/multicandidate_audits/<output_stem>/`

Fichiers produits :

- `all_candidates.csv`
- `accepted_points.csv`
- `rejected_candidates.csv`
- `point_summary.csv`
- `run_config.json`
- `README.md`

Si `--flush-every-candidate` est actif, ces CSV sont reecrits **apres chaque candidat**.

Protection volontaire :

- `--append-validated` **ne modifie pas** `validated_modal_points`
- s'il est active, il ecrit seulement une proposition locale `validated_append_proposal.csv` dans le dossier du run

## Slurm Command

Point unique :

```bash
POINTS="0.200:1.800" \
OUTPUT_STEM=supersonic_multicandidate_M180_a020 \
MAX_CANDIDATES_PER_POINT=12 \
BOX_REQUIRED=1 \
sbatch launch/jz_submit_supersonic_shooting_blumen_locked_multicandidate.slurm
```

Petit bloc :

```bash
POINTS="0.200:1.800 0.225:1.800" \
OUTPUT_STEM=supersonic_multicandidate_M180_a020_0225 \
MAX_CANDIDATES_PER_POINT=12 \
BOX_REQUIRED=1 \
sbatch launch/jz_submit_supersonic_shooting_blumen_locked_multicandidate.slurm
```

Dry-run :

```bash
POINTS="0.150:1.400" \
OUTPUT_STEM=_dryrun_supersonic_multicandidate_M140_a015 \
MAX_CANDIDATES_PER_POINT=6 \
BOX_REQUIRED=1 \
DRY_RUN_CANDIDATES=1 \
sbatch launch/jz_submit_supersonic_shooting_blumen_locked_multicandidate.slurm
```

## Tests Executed

### 1. Python compile

```bash
python3 -m py_compile scripts/audit_supersonic_shooting_blumen_locked_multicandidate.py
```

Resultat : OK

### 2. Slurm shell check

```bash
bash -n launch/jz_submit_supersonic_shooting_blumen_locked_multicandidate.slurm
```

Resultat : OK

### 3. Dry-run lightweight on `M=1.8, alpha=0.2`

```bash
MPLCONFIGDIR=/tmp/mpl_codex_multicandidate python3 \
  scripts/audit_supersonic_shooting_blumen_locked_multicandidate.py \
  --points "0.200:1.800" \
  --anchor-csv assets/classic_supersonic/validated_modal_points/supersonic_validated_modal_points.csv \
  --output-stem _dryrun_supersonic_multicandidate_M180_a020 \
  --max-candidates-per-point 4 \
  --candidate-source auto \
  --dry-run-candidates
```

Resultat :

- execution OK
- CSV et README ecrits
- sur ce point, `auto` n'a trouve que des seeds `existing` de type ancre
- cause constatee : pas de branche guidee brackettable a `M=1.8` et pas de cible `c_i` Blumen interpolable sur ce point

### 4. Dry-run lightweight on `M=1.4, alpha=0.15`

```bash
MPLCONFIGDIR=/tmp/mpl_codex_multicandidate python3 \
  scripts/audit_supersonic_shooting_blumen_locked_multicandidate.py \
  --points "0.150:1.400" \
  --anchor-csv assets/classic_supersonic/validated_modal_points/supersonic_validated_modal_points.csv \
  --output-stem _dryrun_supersonic_multicandidate_M140_a015 \
  --max-candidates-per-point 6 \
  --candidate-source auto \
  --dry-run-candidates
```

Resultat :

- execution OK
- CSV et README ecrits
- `auto` a bien melange plusieurs familles :
  - `anchor_anchor_nearest_2`
  - `branch_guided_interp_cr_target_ci`
  - `blumen_perturb_scale`
  - `anchor_anchor_exact`
  - `branch_guided_interp_branch_ci_blend`
  - `branch_interp_target`

## Remaining Limits

- le vrai smoke test avec shooting complet n'a pas ete lance localement ici, uniquement les dry-runs de generation et de flush
- `M=1.8` reste un cas structurellement difficile : il peut manquer a la fois la contrainte Blumen locale et le bracket branch-guided
- `--append-validated` est volontairement non destructif pour proteger la base de points valides
- le script n'ajoute pas encore de figures PNG/PDF ; il produit d'abord les tables robustes necessaires au tri

## Explicit Safety Confirmation

- aucun fichier dans `assets/classic_supersonic/validated_modal_points/` n'a ete modifie par ce developpement
- aucun point n'est ajoute automatiquement a `supersonic_validated_modal_points.csv`
- le workflow existant `audit_supersonic_shooting_blumen_locked.py` reste intact
