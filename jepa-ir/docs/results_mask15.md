# Résultats — encodeur JEPA-IR, run masquage 15 %

> Toutes les valeurs sont extraites des logs réels (Dalia GPU B200 + éval locale),
> pas estimées.

---

## 0. RÉSULTATS FINAUX (pipeline propre, sans fuite) ⭐

Pipeline rigoureux : encodeur (job `77085`, mask 15 %) entraîné sur le **pool
encoder** (194 798 graphes), predictor chaîne entraîné sur le **pool predictor**
(58 029 programmes, disjoint), évalué sur le **test set jamais vu**.

**Encodeur JEPA** — aucun collapse (cos masqué/complet 0.996, `vs autres` 0.52).

**Predictor chaîne `O0→O1→O2→O3`** (test set held-out) :

| Métrique | Valeur |
|---|---|
| Erreur predictor vs baseline identité | **−50.8 %** (MSE 0.341 → 0.168) |
| dist(O0, O3) (1 − cos) | 0.210 |
| dist(O1, O3) | 0.021 |
| dist(O2, O3) | 0.003 |
| compositionnel : dist(predictor³(O0), O3) | 0.100 (vs 0.210 brut) |

✅ Les 3 propriétés visées : le predictor bat l'identité, les distances décroissent
O0 > O1 > O2 vers O3, et la chaîne rapproche bien de O3.

**Anti-fuite garanti** : 3 pools disjoints par hash (`splits.py`), split
train/val/test 70/15/15, l'encodeur n'a jamais vu les programmes du predictor.

Figures : `runs_from_dalia/figures/` + `runs_from_dalia/o0_o3_5k.png`. Dashboard
de présentation : voir `docs/pitch.md`.

---

## (Historique) — run de référence `76092`

> Rapport du run `76092` (masquage 15 %) et du contexte comparatif des itérations.

## 1. Configuration du run

| Paramètre | Valeur |
|---|---|
| Job Slurm | `76092` (Dalia, nœud B200) |
| Architecture | GNN 3-relations, `hidden=256`, `layers=6` (~2,5 M params) |
| Données | **97 639** graphes (échantillon AnghaBench 100 k) |
| Masquage | `mask_edges=True`, **ratio 15 %** (opcode + arêtes des nœuds masqués) |
| Loss | VICReg, coeffs `sim=1 / std=1 / cov=1` |
| Batch / epochs | 512 / 50 |
| Embedding | dim 128 (le livrable) |

**Loss = VICReg** (pas une simple MSE) : `invariance (MSE masqué↔complet) + variance + covariance`.
Les deux derniers termes interdisent l'effondrement (prouvé par ablation, §4).

## 2. Évolution de la loss

| epoch | loss totale | invariance (MSE) | variance | covariance | emb_std |
|---|---|---|---|---|---|
| 0 | 1.356 | 0.138 | 1.204 | 0.014 | 0.75 |
| 10 | 0.180 | 0.053 | 0.022 | 0.104 | 1.02 |
| 25 | 0.136 | 0.036 | 0.000 | 0.100 | 1.08 |
| **49 (final)** | **0.126** | 0.038 | 0.001 | 0.088 | 1.06 |

- Chute nette ×11 (1.36 → 0.13). Au départ la **variance domine** (1.20, embeddings
  écrasés à std 0.75) ; corrigée en ~5 epochs.
- À convergence, la loss est portée par **invariance (~0.04) + covariance (~0.09)** ;
  la variance est à zéro (std stabilisé à ~1.0, cible du hinge VICReg).

## 3. Anti-collapse (diagnostics tous les 5 epochs)

| epoch | emb_std | rang effectif /128 | \|corr\| | PCA PC1 |
|---|---|---|---|---|
| 0 | 1.011 | 36.7 | 0.210 | 11 % |
| 10 | 1.035 | 33.0 | 0.156 | 7 % |
| 25 | 1.054 | 34.1 | 0.143 | 6 % |
| **final** | 1.014 | **34.9** | **0.136** | **5 %** |

- **Aucun collapse** : `std ≈ 1.0` stable, rang effectif ~35/128, PC1 à seulement 5 %
  (variance étalée, nuage quasi isotrope).
- La corrélation moyenne **baisse** continûment (0.21 → 0.14) : VICReg décorrèle bien.

## 4. Qualité du masquage — `encode(masqué)` vs `encode(complet)`

Mesuré à chaque snapshot (n = 2048, **suivi pendant l'entraînement**) :

| epoch | cos(masqué, complet) | cos vs autres progs | retrieval top-1 |
|---|---|---|---|
| 0 | 0.978 | 0.618 | 51 % |
| 10 | 0.991 | 0.649 | 56 % |
| 25 | 0.992 | 0.577 | 58 % |
| 30 | 0.994 | 0.566 | **59 %** |
| **final** | **0.995** | **0.517** | **58 %** |

- `cos(masqué, complet)` monte à **0.995** : l'encodeur reconstruit presque
  exactement l'embedding complet à partir du graphe masqué → **objectif JEPA atteint**.
- `cos vs autres` **baisse** (0.62 → 0.52) : meilleure séparation des programmes.
- **retrieval top-1 = 58 %** : pour 58 % des programmes, le masqué retrouve *son propre*
  complet parmi 2048 candidats (hasard ≈ 0,05 %).

## 5. Comparaison des runs (le chemin parcouru)

| Run | données | modèle | masquage | retrieval | note |
|---|---|---|---|---|---|
| 75686 | 10 k | hidden128/l4 | doux (opcode seul) | 84 %* | *trivial : ignore les opcodes |
| 75735 | 10 k | hidden128/l4 | **sans VICReg** (ablation) | — | **collapse** (std→0.0003) |
| 75818 | 10 k | hidden128/l4 | edges 30 % | — | corrige : opcodes enfin utilisés |
| 75968 | 100 k | hidden128/l4 | edges 30 % | ~40 % | plateau (capacité limitée) |
| 76061 | 100 k | hidden256/l6 | edges 30 % | ~48 % | modèle plus gros aide |
| **76092** | **100 k** | **hidden256/l6** | **edges 15 %** | **58 %** | **meilleur** |

\* Le 84 % du run 75686 est un **faux bon score** : on a démontré (masquer 100 % des
opcodes → cos 0.999) que l'encodeur ignorait le contenu et ne lisait que la
structure. Le masquage d'arêtes a corrigé ça (cos tombe à 0.48).

### Preuve anti-collapse par ablation (run 75735)

Couper variance + covariance (VICReg → MSE seule) provoque l'effondrement immédiat :
`emb_std 1.07 → 0.0003`, invariance → 0 (encodeur = vecteur constant). Confirme que
les termes variance/covariance sont **indispensables**, pas décoratifs.

## 6. Sensibilité à l'optimisation (-O0 vs -O3)

Mesure sur **4883 programmes** (cosinus, encodeur 76092) :

| Paire | cosinus moyen |
|---|---|
| O0 ↔ O3 **même programme** | **0.740** ± 0.22 |
| O0 ↔ O0 (progs différents) | 0.727 ± 0.15 |
| O3 ↔ O3 (progs différents) | 0.522 ± 0.21 |
| O0 ↔ O3 (progs différents) | 0.586 ± 0.19 |

- Le même programme à O0 et O3 est **la paire la plus proche** (marge +0.15 vs progs
  différents) → l'encodeur **relie** O0 et O3 du même programme.
- Distribution **bimodale** : ~moitié des programmes ont O0≈O3 (cos > 0.85), l'autre
  moitié dans la zone de mélange. Signal réel mais inégal.
- Pas de cluster par niveau d'optimisation (O3 entre eux très dispersés : 0.52).

## 7. Fichiers

| Artefact | Chemin |
|---|---|
| Encodeur entraîné (le livrable) | `runs_from_dalia/job_76092_encoder.pt` (9,4 Mo) |
| Diagnostics PCA (Dalia) | `runs/mask15_76092/diagnostics/pca_epoch*.png` |
| Distribution O0/O3 | `runs_from_dalia/o0_o3_5k.png` |
| Script d'entraînement | `scripts/train.py --mask-edges --mask-ratio 0.15 --hidden-dim 256 --num-layers 6` |
| Probe masquage | `scripts/probe_masking.py`, `scripts/probe_masking_sweep.py` |
| Mesure O0/O3 | `scripts/measure_o0_o3.py` |

## 8. Lecture d'ensemble

L'encodeur (job 76092) produit des représentations **saines** (aucun collapse, rang
~35, dimensions décorrélées) et **utiles** : à partir d'un graphe masqué à 15 %, il
reconstruit un embedding cohérent avec le graphe complet (cos 0.995) et identifie le
bon programme 58 % du temps sur 2048 candidats. C'est le meilleur compromis trouvé
entre difficulté de la tâche (masquage non-trivial) et qualité du signal.