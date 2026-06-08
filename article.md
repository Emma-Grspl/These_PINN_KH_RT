# Working Draft

Tentative title:

`Learning compressible Kelvin-Helmholtz eigenvalues and eigenmodes with physics-informed neural networks: a benchmark against shooting solvers`

This file is no longer only a plan. It now contains a first draft of the opening sections of the paper:

- Introduction
- Physical and spectral problem
- Classical reference solver
- Beginning of the PINN section

The remaining sections are still outlined at the end.

## 1. Introduction

Hydrodynamic and magnetohydrodynamic instabilities remain a central topic in fusion plasma physics because they organize mixing, momentum redistribution, and the transfer of fluctuation energy across scales. Even when the final target is a nonlinear or multi-physics regime, linear stability analysis still provides the first spectral objects from which reduced models, local diagnostics, and surrogate strategies can be built. In that sense, reliable eigenvalue solvers are not only diagnostic tools: they are also reusable numerical components for larger modelling pipelines.

The present work focuses on a compressible Kelvin-Helmholtz shear layer with a smooth base profile. This configuration is compact enough to stay interpretable and at the same time rich enough to benchmark a spectral learning workflow: the eigenvalue is complex, the modal structure depends on asymptotic branch selection, and nearby modal families can coexist over part of parameter space. These features make the problem well suited to assessing whether a physics-informed neural network can learn a full spectral object rather than only a single scalar observable.

Physics-informed neural networks, or PINNs, are attractive in that setting because they approximate parameter-dependent solutions while enforcing the governing equations directly inside the loss function. This opens the possibility of learning a map from control parameters to eigenvalues and eigenmodes without assembling a dense classical database over the whole domain. For spectral problems, however, the target of the learning task is composite: one must recover both the admissible eigenvalue and the associated eigenfunction, together with a consistent branch identity across parameter space. The present paper therefore treats the PINN and the classical solver as complementary ingredients of the same benchmark.

Our methodology starts from local shooting solvers in subsonic and supersonic regimes. These solvers are used to identify which points are spectrally and modally validated, which continuations remain branch-consistent, and which regions currently support only spectral validation. Once this local reference is organized, we use it to benchmark PINNs with metrics that distinguish scalar spectral accuracy from modal reconstruction quality. This separation is central to the paper: learning a growth rate such as \(c_i\) is one task, whereas learning a branch-consistent eigenmode is another.

The paper is organized as follows. Section 2 introduces the compressible shear layer and the associated pressure eigenvalue problem. Section 3 presents the classical shooting framework used as local reference in both subsonic and supersonic regimes, including the boundary conditions and coordinate mappings used in practice. Section 4 introduces the PINN formulation adopted in this work, summarizes the benchmark configurations that were tested, and defines the role of sparse classical guidance. The remaining sections will then report the subsonic and supersonic benchmarks and discuss what the present workflow already validates for spectral PINNs.

## 2. Physical and Spectral Problem

### 2.1. Compressible shear layer

We consider a two-dimensional inviscid compressible shear layer with base velocity

\[
U(y) = \tanh(y). \tag{1}
\]

Equation (1) is the smooth mixing-layer profile used in the classical compressible Kelvin-Helmholtz literature. The streamwise direction is denoted by \(x\), the transverse direction by \(y\), and the perturbations are parameterized by a real streamwise wavenumber \(\alpha\) and a Mach number \(M\), where \(M\) denotes the ratio between the characteristic flow velocity and the sound speed of the reference state.

In the nondimensional formulation used here, the base flow is parallel and reads

\[
\bigl(\bar u,\bar v,\bar p,\bar \rho\bigr)(y) = \bigl(U(y),0,\bar p_0,\bar \rho_0\bigr), \tag{2}
\]

where \(\bar p_0\) and \(\bar \rho_0\) are uniform reference thermodynamic states and only the streamwise velocity varies across the shear layer through Eq. (1). The unknowns of the spectral problem are therefore perturbations around the base state in Eq. (2).

The same spectral equation is used throughout the paper for all values of \(M\). In that sense, there is no mathematical separation between the subsonic and supersonic cases. The distinction introduced below is physical and numerical: the asymptotic modal structure induced by the same equation changes with \(M\), and the reference workflow is organized accordingly.

### 2.2. Normal-mode formulation

The total state is decomposed as base flow plus perturbation,

\[
\bigl(u,v,p,\rho\bigr)(x,y,t)
=
\bigl(\bar u,\bar v,\bar p,\bar \rho\bigr)(y)
+
\bigl(u',v',p',\rho'\bigr)(x,y,t), \tag{3}
\]

and the perturbation fields are sought under the standard normal-mode ansatz

\[
q'(x,y,t) = \hat q(y)\,\exp\!\bigl(i\alpha(x-ct)\bigr), \qquad c = c_r + i c_i. \tag{4}
\]

The temporal growth rate is therefore

\[
\omega_i = \alpha c_i. \tag{5}
\]

Equation (4) shows that the unknown is a spectral pair composed of the complex phase velocity \(c\) and the transverse eigenfunction \(\hat q(y)\). Equation (5) identifies \(c_i\) as the quantity that controls temporal amplification, while \(c_r\) sets the phase velocity.

After elimination of the other perturbation variables, the problem can be written in terms of a pressure-like scalar amplitude \(\hat p(y)\) satisfying

\[
\hat p'' - \frac{2U'}{U-c}\hat p' - \alpha^2 \Bigl[1 - M^2(U-c)^2\Bigr]\hat p = 0. \tag{6}
\]

Equation (6) is the pressure form used throughout the paper. It concentrates the spectral problem into a single transverse unknown and provides direct access to the far-field structure through the coefficients of the ODE.

### 2.3. Spectral interpretation

The rest of the paper uses Eq. (6) in two complementary ways. First, it provides a pointwise spectral equation for the classical shooting solvers. Second, it serves as the residual backbone of the PINN formulations. The benchmark therefore keeps the same mathematical object on both sides and changes only the numerical representation: direct shooting in the classical reference, neural approximation in the PINN.

For the present benchmark, the key quantity is not only the scalar growth rate from Eq. (5), but the full pair \((c,\hat p)\). The validation protocol is organized around that observation: a point may be spectrally acceptable in terms of \(c\) and still require a separate modal check on \(\hat p\) and on the other reconstructed fields \((\hat \rho,\hat u,\hat v)\).

## 3. Classical Reference Solver

### 3.1. Reference strategy

Before training a PINN, we first construct a local classical reference. This step is methodological rather than cosmetic. Since the same pressure equation supports several nearby modal families over part of the \((\alpha,M)\) plane, the benchmark requires a solver that treats a single parameter pair at a time, imposes the admissible far-field behavior, and exposes diagnostics on both the eigenvalue and the mode.

The reference used in this work is based on shooting methods. The common idea is the following: propagate admissible far-field branches toward a matching point, then adjust the eigenvalue until the left and right solutions become compatible. In practice, the production workflow combines:

- a fast subsonic Riccati shooting solver for dense local sweeps;
- a more complete subsonic `mstab17`-style cross-check near neutrality and for modal reconstruction;
- a two-stage supersonic Riccati shooting solver for pointwise and short-line validation.

This organization reflects what was actually tested in the repository. The classical benchmark is first established pointwise, then extended through validated local continuations. Global reconstructions are built only from points that already satisfy local spectral and modal checks.

### 3.2. Riccati reformulation

The pressure equation in Eq. (6) is rewritten through the Riccati variable

\[
\gamma(y) = \frac{\hat p'(y)}{\hat p(y)}. \tag{7}
\]

Substituting Eq. (7) into Eq. (6) gives the first-order nonlinear equation

\[
\gamma' = -\gamma^2 - P(y)\gamma + \alpha^2 R(y), \qquad
P(y) = -\frac{2U'(y)}{U(y)-c}, \qquad
R(y) = 1 - M^2\bigl(U(y)-c\bigr)^2. \tag{8}
\]

Equation (8) is the working form of all the shooting solvers used in the present paper. It turns the far-field selection into a condition on asymptotic values of \(\gamma\), and it replaces raw pressure matching by a local matching condition at a prescribed transverse location.

### 3.3. Boundary conditions and coordinate mappings

The physical boundary condition is posed on the unbounded domain \(y\in\mathbb{R}\). On the left and right far fields, Eq. (6) is evaluated in the uniform limits \(U(-\infty)=-1\) and \(U(+\infty)=+1\). The admissible branch is then selected by the asymptotic behavior

\[
\hat p(y) \sim A_- e^{\gamma_- y}\quad (y\to -\infty),\qquad
\hat p(y) \sim A_+ e^{\gamma_+ y}\quad (y\to +\infty), \tag{9}
\]

with \(\Re(\gamma_-)>0\) and \(\Re(\gamma_+)<0\). In the subsonic regime, Eq. (9) produces decaying branches on both sides. In the supersonic regime, the same condition may combine decay and oscillation through the imaginary parts of \(\gamma_\pm\), while keeping the real-part sign convention that selects the outgoing admissible branch.

All practical solvers truncate the physical domain after this branch selection step. Two coordinate choices are used in the present work:

- direct integration in the physical variable \(y\);
- an algebraic compactification

\[
y = L_\xi \frac{\xi}{1-\xi^2}, \qquad \xi\in(-1,1), \tag{10}
\]

where \(L_\xi\) is a mapping scale.

The fast subsonic solver and the subsonic `mstab17` cross-check both integrate directly in \(y\) on a truncated interval \([-L,L]\), where \(L\) is estimated from the asymptotic decay rates. The supersonic single-case and modal-validation solver uses the same Riccati equations and the same asymptotic branch condition, with Eq. (10) activated in the mapped runs used for pointwise audit and modal reconstruction. The same algebraic mapping is also reused in the PINN.

### 3.4. Subsonic workflow

In the subsonic branch considered here, the unstable phase velocity is purely imaginary, so the eigenvalue search reduces to

\[
c = i c_i, \qquad c_i \ge 0. \tag{11}
\]

Equation (11) collapses the search to one scalar degree of freedom. The fast subsonic solver integrates Eq. (8) directly in \(y\) from the two far fields toward a matching point \(y_m=0\), and it minimizes the Riccati mismatch

\[
\mathcal{J}(c_i) = \left|\gamma_L(y_m)-\gamma_R(y_m)\right|. \tag{12}
\]

The subsonic production reference is not based on a single implementation only. In most of the parameter range, the scalar mismatch in Eq. (12) is sufficient and gives an efficient reference. Near neutrality, the workflow is complemented by a more complete `mstab17`-style solver that propagates the real Riccati state \([\kappa,q,\ln|\hat p|,\phi]\), first matches \((\kappa,q)\) at a positive matching point, and then aligns the reconstructed pressure amplitude near the layer center. Both solvers use the same pressure equation, the same far-field branch condition from Eq. (9), and the same direct \(y\)-space truncation; they differ only in the richness of the matching diagnostics.

This is the reason why the subsonic reference is the first benchmark window used for the PINN. It combines a dense local reference for \(c_i\) with a branch-consistent modal cross-check on the same mathematical formulation.

### 3.5. Supersonic workflow

In the supersonic regime, the same pressure equation is retained and the branch selection still comes from Eq. (9). The practical change is that the search is now performed on a genuinely complex phase velocity \(c=c_r+i c_i\), and the local state is represented through

\[
\gamma = \kappa + i q, \qquad
\bigl[\kappa,\;q,\;\ln|\hat p|,\;\phi\bigr]. \tag{13}
\]

The pointwise supersonic solver used in this work is organized in two stages:

1. a spectral stage that matches the Riccati variables \((\kappa,q)\) at the matching point and identifies \((c_r,c_i)\);
2. an amplitude stage that adjusts the right-branch initialization so that the reconstructed pressure mode is aligned near the center.

The solver is implemented both in direct \(y\) coordinates and, for the audited modal runs used in the present benchmark, with the mapped coordinate from Eq. (10). This two-stage structure is also the basis of the local spectral and modal audit used later in the paper: one first validates the eigenvalue, then validates the reconstructed mode.

This distinction is important for the dataset organization. Some supersonic points are retained as modal references, while others are retained only as spectral references. The present paper therefore keeps a sharp separation between:

- pointwise validated modal anchors;
- spectral-only validated points;
- global reconstructions obtained from short validated continuations.

### 3.6. Role of the classical reference in this paper

The shooting solver is used here as a trusted local reference, not as a black-box generator of dense maps. This choice directly informs the PINN study. A neural benchmark is first asked to reproduce local classical solutions whose spectral and modal consistency has already been checked. The trusted training and validation domains are therefore delimited by the classical audit itself.

This benchmark logic also clarifies what is and is not tested at this stage. We do test local subsonic and supersonic reference points, local line continuations, modal densification around validated anchors, and branch-consistency diagnostics. We do not assume that every point of a dense supersonic map is already modally validated. That is precisely why the PINN study is organized progressively.

## 4. Physics-Informed Neural Networks for the Spectral Benchmark

### 4.1. PINN principle

A physics-informed neural network is a neural approximation of an unknown function in which the governing equations are enforced during training. Instead of learning only from paired input-output data, the network is trained by minimizing a loss function that contains residual terms derived from the differential equations, together with boundary, normalization, matching, or auxiliary constraints.

For the present problem, the network does not only represent a field. It must represent a spectral pair: an eigenvalue branch and a modal branch. The spectral branch returns the admissible phase velocity, while the modal branch returns the transverse structure associated with that phase velocity. The training objective therefore combines local differential admissibility, asymptotic branch consistency, normalization, and, when required, sparse classical guidance.

![Schematic view of the PINN workflow used in the benchmark. Inputs are the mapped transverse coordinate and the physical parameters. The network is split into a spectral branch and a modal branch; automatic differentiation provides the residual terms, and optional sparse classical anchors stabilize branch selection when needed.](assets/article/pinn_spectral_workflow.svg)

Figure 1 summarizes this workflow.

### 4.2. PINN parameterization used in this work

The present study starts from the same pressure formulation as the classical reference. In the fixed-Mach subsonic benchmarks, the network inputs are the mapped transverse coordinate \(\xi\) and the wavenumber \(\alpha\), with the algebraic mapping from Eq. (10). The physical coordinate is therefore not learned implicitly: it is prescribed explicitly through the same compactification used in the classical modal solver.

Two neural outputs are then distinguished:

- a spectral branch, which returns \(c_i(\alpha)\) in the subsonic fixed-Mach benchmarks;
- a modal branch, which returns either pressure components or Riccati components depending on the representation.

In the benchmarks carried out so far, the Riccati representation is central because it matches the classical reference organization. The network predicts \((\kappa,q)\), then reconstructs pressure through the inverse use of Eq. (7). This choice turns the asymptotic information into boundary-band constraints on the Riccati variables and keeps the neural output aligned with the quantities used in the shooting solver.

The present paper therefore emphasizes pressure-based and Riccati-based parameterizations rather than a direct four-field regression everywhere in the domain. Full-field supervision is used only as a targeted auxiliary ingredient in selected experiments.

### 4.3. Loss construction

The PINN loss is built around the same spectral equation as the classical reference. Depending on the chosen parameterization, the residual term is formed either from Eq. (6) itself or from its Riccati form in Eq. (8). Automatic differentiation supplies the derivatives in the mapped coordinate, and the chain rule is applied through Eq. (10).

On top of the residual term, the benchmark uses four families of auxiliary constraints:

- asymptotic or boundary-band constraints derived from the far-field branch condition in Eq. (9);
- normalization and phase-fixing constraints for the reconstructed mode;
- matching constraints on Riccati variables or reconstructed pressure;
- optional sparse classical supervision on spectral or modal quantities.

This organization mirrors the structure of the classical solver rather than replacing it by an unrelated neural objective. In particular, the PINN is asked to satisfy the same pressure equation, the same asymptotic branch logic, and the same distinction between spectral admissibility and modal consistency.

### 4.4. Sparse guidance versus dense supervision

The guiding principle of the benchmark is to maximize the role of physics-based training while keeping classical supervision as light as possible and as targeted as necessary. In practice, the present study treats the different forms of guidance hierarchically:

- sparse spectral supervision, for instance on \(c_i\), is the lightest form of classical information;
- sparse Riccati or pressure anchors provide local modal orientation when the branch identity benefits from additional stabilization;
- dense field supervision is reserved for specific ablation or repair experiments.

This hierarchy is motivated by the structure of the problem itself. A scalar constraint on the spectral branch is inexpensive and often highly informative, whereas dense field supervision is more restrictive and is used only where it improves modal continuity in a controlled way. The benchmark is therefore hybrid by construction: the physics residual remains the backbone of the training, and classical information is added only in forms that can be ablated and interpreted.

### 4.5. What was tested in the present study

The benchmark sequence follows the current state of validation of the classical reference.

First, the PINN is tested in a one-dimensional fixed-Mach subsonic setting. This is the most controlled benchmark window because the classical reference is dense, the pressure equation in Eq. (6) is validated there branch by branch, and modal reconstruction can be audited against local shooting solutions. Within that window, the study includes:

- joint reconstruction of the spectral branch and the modal branch over a fixed-Mach interval in \(\alpha\);
- sparse Riccati guidance and sparse \(q\)-/\(\gamma\)-based supervision;
- targeted edge-focused modal repair over a reduced \(\alpha\)-window with a frozen spectral branch.

Second, the benchmark is extended conceptually toward two-parameter subsonic learning in \((\alpha,M)\), while keeping the one-dimensional fixed-Mach study as the first validated layer of the methodology.

Third, the supersonic study is organized around the validated classical reference rather than around a dense PINN regression target. In other words, the supersonic benchmark uses pointwise and linewise validated anchors to define the region where a PINN comparison is meaningful. A full dense supersonic PINN on the whole \((\alpha,M)\) domain is therefore not the starting point of the present paper; the reference is first stabilized locally and then used to delimit the benchmark window.

### 4.6. Validation metrics

The validation protocol always distinguishes spectral accuracy from modal accuracy.

For the spectral part, the natural scalar metrics are errors on \(c_i\), and on \(c_r\) whenever the reference branch is genuinely complex. For the modal part, the benchmark uses field-based discrepancies, envelope and phase diagnostics, branch-continuity checks, and visual overlays against the local classical reference.

This separation is one of the central methodological points of the paper. A PINN may be spectrally accurate in the sense of Eq. (5) while still benefiting from additional modal guidance. Conversely, a mode may be visually coherent only if it is attached to the correct spectral branch. The benchmark is therefore evaluated through both pieces of information at once.

## Proposed full paper plan

The paper should answer one central question:

> To what extent can a PINN solve a non-selfadjoint compressible eigenvalue problem, that is, recover both the unstable eigenvalue and the physically relevant eigenmode over parameter space?

For a physics audience, the paper must remain explicit about three distinct targets:

1. recovering a scalar spectral quantity such as \(c_i\);
2. recovering a full modal structure;
3. preserving the correct branch identity when \((\alpha,M)\) varies.

These three targets should not be merged. A large part of the scientific interest of the paper is precisely to show that they do not have the same difficulty level.

### Editorial line to keep throughout the paper

The paper should defend the following message:

- the compressible KH equation is the same in subsonic and supersonic regimes;
- however, the numerical reference and the PINN benchmark do not have the same level of difficulty in both regimes;
- the right question is therefore not "does a PINN work or fail globally?";
- the right question is "which part of the spectral problem can be learned reliably, under which constraints, and with which level of classical guidance?"

This framing justifies the split between subsonic and supersonic sections without pretending that the governing equation changes.

### What Section 4 must explain for a physics audience

Because the target readership is not assumed to know PINNs well, Section 4 should not stay too compact. It should explicitly explain:

- what a PINN is in operational terms:
  - inputs;
  - outputs;
  - collocation points;
  - automatic differentiation;
  - residual-based loss;
  - optimization;
- why eigenvalue problems are harder than standard forward PINN problems:
  - the unknown is a pair \((c,\hat p)\), not only a field;
  - the mode is defined up to normalization and phase;
  - several nearby spectral branches may coexist;
  - low residual does not automatically imply correct branch selection;
- why this work uses two neural branches:
  - a spectral branch for \(c_i\) or \((c_r,c_i)\);
  - a modal branch for the transverse structure;
- why the benchmark separates spectral and modal validation;
- what "sparse classical guidance" means in practice and why it is used.

If needed, Section 4 can be expanded with one additional subsection:

- `4.7 From local eigenpairs to a parameterized PINN map`

This subsection would explain the progressive strategy:

- local eigenpair recovery;
- fixed-Mach branch learning;
- narrow-band \((\alpha,M)\) learning;
- later globalization.

That would make the training logic much clearer to non-specialists of PINNs.

## 5. Subsonic PINN benchmark

This section should be the first strong positive result of the paper. It is where the reader should become convinced that the methodology works on a controlled benchmark window.

### 5.1. Why start with the subsonic regime

Main message:

- same governing equation as before;
- but the subsonic reference is denser, cleaner, and more branch-consistent;
- this makes it the natural first benchmark window for evaluating whether a PINN can recover eigenpairs.

### 5.2. Local and fixed-Mach benchmark setup

Explain clearly what is benchmarked first:

- fixed-Mach lines;
- local classical references at `M = 0.5` and `M = 0.6`;
- comparison metrics on:
  - \(c_i\);
  - pressure mode;
  - reconstructed \(\hat \rho\), \(\hat u\), \(\hat v\);
  - amplitude and phase errors.

This subsection should establish the benchmark protocol before showing results.

### 5.3. What pure physics recovers locally, and what it does not

This subsection is important scientifically. It should not be hidden.

Main message:

- a pure-PINN local eigenpair is a meaningful target;
- however, asking a PINN to learn a full fixed-Mach branch `c_i(\alpha)` from scratch with physics only is too ambitious in the present formulation;
- the negative fixed-Mach run is therefore an ablation result, not a failure to be buried.

This is where the negative run around job `1936761` belongs:

- not as the headline result;
- but as evidence that branch learning is harder than local eigenpair recovery.

### 5.4. Hybrid fixed-Mach branch learning

This subsection should present the successful fixed-Mach subsonic runs:

- `M = 0.5` reference;
- `M = 0.6` core reconstruction;
- `M = 0.6` edge-focused modal repair.

Main message:

- with the present loss design, the PINN can recover accurate subsonic growth rates and good modal structures over validated \(\alpha\)-intervals;
- sparse guidance is especially useful for stabilizing the modal branch without destroying the spectral one.

### 5.5. First two-parameter subsonic map

This is the place for the `M \in [0.5,0.6]` pilot band, and possibly the first extension toward `0.7`.

Main message:

- a parameterized PINN can already learn a nontrivial \((\alpha,M)\) dependence;
- interpolation at intermediate Mach is physically meaningful;
- extension in Mach is possible but must be stabilized to avoid degrading already learned modal content.

This subsection is where the paper should first show the final target interface:

- input: \((\xi,\alpha,M)\) or \((\alpha,M)\);
- output: mode and spectral quantity.

### 5.6. Subsonic synthesis

End the section with a short synthesis that answers explicitly:

- what has been validated for eigenvalues;
- what has been validated for modes;
- what remains out of scope at this stage.

Recommended conclusion of the section:

- local eigenpairs: yes;
- fixed-Mach branches: yes, with sparse guidance;
- narrow-band \((\alpha,M)\) map: yes, in a controlled window;
- global pure-physics branch learning from scratch: not yet.

## 6. Supersonic PINN benchmark

This section should not mimic the subsonic one mechanically. Its role is different: first delimit the trustworthy benchmark window, then evaluate the PINN only where the reference is meaningful.

### 6.1. Why the supersonic regime is harder

This subsection should make the difficulty explicit, without turning into a literature review digression.

Main message:

- same pressure equation;
- but more delicate asymptotic behavior;
- coexistence of nearby modal families;
- weaker decay and stronger branch ambiguity;
- modal continuity is therefore much harder to establish than spectral continuity.

This is where the distinction between `gold`, `silver`, and `branch-ambiguous` reference points should be introduced in paper form.

### 6.2. Construction of a usable supersonic reference

Present the actual reference hierarchy:

- `gold`: locally validated eigenvalue and mode;
- `silver`: reliable spectral point without full modal validation;
- `unresolved / branch ambiguous`: not used as hard benchmark points.

Main message:

- a continuous modal surface is not required for the benchmark to be scientifically usable;
- what matters is a trustworthy pointwise and linewise reference with explicit confidence levels.

### 6.3. Supersonic benchmark windows

This subsection should define where the PINN is allowed to be compared:

- spectral benchmark on `gold + silver`;
- modal benchmark only on `gold`;
- short line continuations only where the branch identity is already controlled.

This is the right place to present the current validated or near-validated Mach windows:

- stable anchors around `M = 1.2`, `1.3`, `1.5`;
- candidate transition zone near `M = 1.4`;
- extension targets toward `M = 1.1` and `M = 1.6-1.8`.

### 6.4. Supersonic PINN tests

This subsection should report the actual supersonic PINN experiments when they are ready. The right order is:

1. local pointwise PINN against `gold` points;
2. short linewise PINN tests on branch-consistent windows;
3. only then broader \((\alpha,M)\) generalization.

Main message:

- in supersonic flow, the neural question must be subordinated to branch validity of the reference;
- a PINN can only be judged where the classical benchmark is itself trusted.

### 6.5. Supersonic synthesis

The section should end by answering:

- what is already validated spectrally;
- what is already validated modalement;
- what remains only exploratory.

The important point is to avoid overstating continuity where the reference itself is branch-ambiguous.

## 7. Discussion

This section should be written as the answer to the main scientific question, not as a loose recap.

### 7.1. What a PINN can recover in a spectral hydrodynamic problem

Main message:

- PINNs can recover meaningful local eigenpairs;
- they can also learn controlled parameterized maps when the reference geometry is sufficiently stable;
- they are not automatically robust branch selectors in non-selfadjoint spectral problems.

### 7.2. What remains difficult

This subsection should isolate the hard points:

- branch competition;
- weakly decaying or radiative supersonic modes;
- mismatch between low residual and correct spectral identity;
- instability of global pure-physics branch regression.

### 7.3. Role of classical guidance

This part should be very clear, because it is likely to matter to reviewers.

Main message:

- the point of the paper is not to hide classical guidance;
- it is to measure how much guidance is needed and where;
- sparse spectral guidance and sparse modal anchors are often much more efficient than dense field supervision;
- the amount of required guidance depends strongly on the spectral regime.

### 7.4. Implications beyond Kelvin-Helmholtz

Broaden the message carefully:

- the workflow is relevant for other non-selfadjoint eigenvalue problems;
- especially where eigenvalues and eigenfunctions must be learned jointly and branch identity matters;
- the paper therefore contributes not only a KH benchmark, but a methodology for spectral PINNs.

## 8. Conclusion

The conclusion should be short and concrete.

It should restate three levels of achievement:

1. what has been shown at the local eigenpair level;
2. what has been shown at the parameterized-map level;
3. what still requires stronger reference construction or guidance.

Recommended final message:

- the paper does not claim that a PINN can replace every classical eigenvalue solver from scratch;
- it shows more precisely under which conditions a PINN can reconstruct compressible KH eigenvalues and eigenmodes;
- and it identifies where classical branch-resolved reference information remains essential.

## Suggested figure logic

The paper should probably be organized around a small number of high-value figures instead of many similar overlays.

Suggested minimal set:

1. physical setup + definition of \((\alpha,M,c)\);
2. PINN workflow schematic;
3. classical subsonic reference example:
   - \(c_i(\alpha)\) line;
   - one or two modal reconstructions;
4. subsonic fixed-Mach PINN benchmark:
   - \(c_i\) error;
   - modal overlays;
5. first 2D subsonic band:
   - learned \(c_i(\alpha,M)\);
   - error heatmap;
6. supersonic reference hierarchy:
   - `gold`, `silver`, `ambiguous`;
7. supersonic local or linewise PINN comparisons on trusted windows;
8. summary figure comparing what is validated:
   - local eigenpair;
   - branch continuation;
   - global parameterized learning.

## Suggested table logic

At least three compact tables would help:

1. classical reference status by regime:
   - subsonic dense lines;
   - supersonic gold/silver counts;
2. subsonic PINN metrics:
   - \(c_i\) errors;
   - pressure and modal errors;
3. supersonic PINN metrics on trusted windows:
   - spectral metrics on `gold + silver`;
   - modal metrics on `gold`.

## Claims to defend, and claims to avoid

Claims to defend:

- PINNs can solve local compressible KH eigenvalue problems;
- branch-consistent modal reconstruction is harder than scalar spectral recovery;
- subsonic and supersonic regimes require different benchmark logic even when the governing equation is the same;
- sparse classical guidance can be an efficient stabilizer rather than a contradiction of the PINN approach.

Claims to avoid unless the final results really support them:

- a fully unsupervised global \((\alpha,M)\) PINN from scratch over the whole domain;
- a continuous and fully validated supersonic modal surface everywhere;
- the idea that low physics residual alone guarantees correct spectral branch selection.
