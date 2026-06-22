# Analyse par transition du predictor — et pourquoi O2→O3 dégrade

> Document d'analyse. À lire avant de présenter le chiffre global du predictor :
> le « +50.8 % » moyen est **trompeur**. Voici la vraie image, transition par
> transition, et l'explication du phénomène.

## 1. Le constat (validation par transition, test set held-out)

8 677 programmes du **test set** (pool predictor, jamais vus). Pour chaque
transition, on compare l'embedding prédit à la vraie cible :

| Transition | MSE identité | MSE predictor | gain | cos(input, cible) | cos(prédit, cible) |
|---|---|---|---|---|---|
| **O0 → O1** | 0.912 | 0.393 | **+56.9 %** | 0.795 | **0.915** |
| **O1 → O2** | 0.094 | 0.085 | +9.1 % | 0.982 | 0.983 |
| **O2 → O3** | 0.018 | 0.025 | **−36.3 %** | 0.997 | 0.995 |

- **MSE identité** = erreur si on recopie l'entrée sans rien prédire. Grande =
  transition *difficile* (input loin de la cible).
- **O0→O1** est le seul vrai saut (input à cos 0.795 de la cible) — et le predictor
  y excelle : cos 0.795 → **0.915**.
- **O2→O3** : l'input est déjà à cos **0.997** de la cible → il n'y a quasi rien à
  prédire, et le predictor *dégrade* légèrement (0.997 → 0.995).

**Le +50.8 % global mélangeait ces trois cas** : une vraie réussite (O0→O1) noyée
avec une transition triviale où le predictor fait pire. Le chiffre moyen
survendait la capacité réelle.

## 2. Ce phénomène N'EST PAS un collapse VICReg

⚠️ Terminologie : il ne faut PAS confondre avec l'effondrement de l'espace latent.

- **Collapse VICReg** (qu'on a *évité*, prouvé par ablation) : l'encodeur sortirait
  un vecteur quasi constant (std → 0). Ici **l'espace est sain** : std ~1.3, rang
  effectif ~30/128. Pas de collapse.
- **Le phénomène observé ici** : les embeddings des niveaux **O1, O2, O3 sont
  quasi-confondus** entre eux (distances à O3 : O1=0.021, O2=0.003). Ce n'est pas
  un effondrement de tout l'espace, c'est que **ces trois niveaux occupent presque
  le même point** pour un programme donné.

On appelle ça ici une **quasi-confusion des niveaux d'optimisation élevés**, pas un
collapse.

## 3. Pourquoi O1 ≈ O2 ≈ O3 (deux explications, non exclusives)

**(a) clang sature tôt.** L'essentiel des optimisations qui changent la *structure*
du programme (mem2reg, SSA, simplifications) est appliqué dès **-O1**. Passer de O1
à O2 puis O3 ajoute surtout de l'inlining/vectorisation fine qui modifie peu le
graphe 3-relations. Donc, sémantiquement, O1/O2/O3 *se ressemblent vraiment*.

**(b) L'encodeur (entraîné sur -O1) lisse les différences.** L'encodeur JEPA n'a vu
que du **-O1** à l'entraînement. Sur O1/O2/O3 il est en terrain familier et produit
des embeddings proches ; les fines différences O2 vs O3 ne sont peut-être pas
captées.

## 4. Pourquoi le predictor unique DÉGRADE sur O2→O3

Notre predictor est **un seul MLP** entraîné sur les **3 transitions mélangées**. Or
elles demandent des comportements opposés :

| Transition | déplacement latent attendu |
|---|---|
| O0→O1 | **grand** (~0.21) |
| O1→O2 | petit (~0.018) |
| O2→O3 | quasi nul (~0.003) → l'**identité** |

Le predictor est **résiduel** (`sortie = entrée + Δ`). Entraîné sur le mélange, il
apprend un Δ « moyen » non-nul. Pour O2→O3, le bon Δ est ≈ 0 ; il en applique un
trop grand → il **dépasse la cible** → cos 0.997 → 0.995. Un seul réseau ne peut pas
être à la fois « bouge beaucoup » (O0→O1) et « ne bouge pas » (O2→O3).

## 5. Pistes de correction (à décider)

1. **Un predictor par transition** (3 MLP). Chacun apprend sa difficulté ; celui de
   O2→O3 apprendra ~l'identité. Reste compositionnel.
2. **Predictor conditionné par le niveau** : un seul MLP qui reçoit le niveau
   d'entrée (one-hot) → peut adapter son Δ. Garde un modèle unique.
3. **Se concentrer sur O0→O1** : assumer que c'est la seule transition non-triviale
   (clang sature dès O1) et présenter ce résultat fort (+57 %, cos 0.79→0.92).

## 6. Ce qu'il faut RETENIR pour la présentation

- ✅ **Résultat fort et honnête** : sur le saut difficile O0→O1, le predictor
  reconstruit l'embedding optimisé avec cos **0.915** (vs 0.795 sans lui).
- ✅ **Résultat scientifique annexe** : O1/O2/O3 sont quasi-confondus dans le latent
  → clang fait l'essentiel du travail structurel dès -O1.
- ❌ **À NE PAS présenter** : le « +50.8 % » global seul — il est gonflé par les
  transitions triviales et masque la dégradation sur O2→O3.

*Chiffres extraits de `scripts/eval_predictor_per_step.py` sur encodeur 77085 +
predictor_chain_final, test set du pool predictor (8 677 programmes).*