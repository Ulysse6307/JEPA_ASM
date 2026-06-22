# JEPA-IR — argumentaire (jury technique)

> Encodeur auto-supervisé de programmes + prédiction latente de l'optimisation
> compilateur. Pitch pour un public qui connaît JEPA / GNN / compilation.

---

## 1. Le problème (compilation)

Décider **quelles optimisations appliquer** à un programme est un problème ouvert et
coûteux :

- Pour savoir si une séquence d'optimisations accélère un code, il faut le
  **compiler ET l'exécuter** — répété sur des milliers de variantes, c'est
  prohibitif (le « phase-ordering problem »).
- Les heuristiques des compilateurs (`-O1/-O2/-O3`) sont des compromis figés, pas
  adaptés au code particulier.

**Il manque une représentation du programme sur laquelle raisonner *avant*
d'exécuter.**

## 2. Notre proposition

```
code  →  LLVM IR  →  graphe 3-relations (sans perte)  →  encodeur GNN (JEPA)  →  embedding
                                                                                    │
                                                            predictor latent ───────┘
                                                            emb(O_k) → emb(O_{k+1})
```

Deux contributions :

**(a) Un encodeur auto-supervisé de programmes.** On apprend, *sans aucun label*,
à transformer un programme en vecteur capturant sa sémantique. L'entrée n'est ni un
AST ni un simple CFG mais un **graphe d'IR qui conserve simultanément les trois
relations** : flot de contrôle, flot de données, **et ordre des effets mémoire** —
ce que les représentations existantes (ProGraML, inst2vec) ne font pas toutes.

**(b) Une prédiction latente de l'optimisation.** Un predictor apprend à
« avancer d'un cran d'optimisation » *dans l'espace latent* :
`predictor(emb(O0)) ≈ emb(O1)`, etc. Appliqué en chaîne, il prédit l'effet de
l'optimisation **sans recompiler ni exécuter**.

## 3. Pourquoi JEPA (et pas un autoencodeur / un décodeur)

- **Pas de reconstruction, pas de décodeur** : la cible est un *vecteur*, jamais le
  graphe. On évite le coût et les artefacts de la génération de code.
- Le signal d'apprentissage vient du **masquage** : on cache une partie du graphe
  (nœuds + arêtes) et on force l'encodeur à produire un embedding cohérent avec le
  graphe complet. C'est ce qui l'oblige à apprendre la *structure réelle* du
  programme, pas sa surface.
- **Anti-collapse explicite (VICReg)** : variance + covariance interdisent la
  solution dégénérée « encodeur → vecteur constant ».

## 4. Ce qu'on a démontré (résultats)

**L'apprentissage marche et ne s'effondre pas** (run de référence) :
- rang effectif de l'espace latent : 37 → **50 / 128** au cours de l'entraînement
  (s'enrichit, opposé d'un collapse) ; PC1 = 4 % (nuage isotrope).
- **Ablation décisive** : couper VICReg fait s'effondrer l'espace
  (`emb_std 1.07 → 0.0003`). Preuve par contraste que la régularisation est ce qui
  empêche le collapse.

**Le masquage est non-trivial** : on a découvert qu'un masquage du seul *contenu*
laissait l'encodeur tricher via la topologie (masquer 100 % des opcodes → cos 0.999).
En masquant **aussi les arêtes**, l'encodeur est forcé d'utiliser le contenu
(cos tombe à 0.48) → représentation réellement informative.

**La prédiction d'optimisation fonctionne** (predictor chaîne, encodeur `77085` +
58 029 programmes du pool predictor, évalué sur le **test set jamais vu**) :
- bat la baseline identité de **−50.8 %** d'erreur (MSE latente 0.341 → 0.168).
- **ordre des distances respecté** : `dist(O0,O3)=0.210 > dist(O1,O3)=0.021 >
  dist(O2,O3)=0.003` — plus un code est optimisé, plus son embedding est proche de
  l'optimum.
- **compositionnalité** : appliquer le predictor 3× sur `emb(O0)` rapproche
  effectivement de `emb(O3)` (distance 0.210 → 0.100).

**Méthodologie anti-fuite** : corpus découpé en 3 pools disjoints par hash
(encodeur / predictor / held-out), split train/val/test déterministe. L'encodeur
n'a jamais vu les programmes qui évaluent le predictor.

## 5. Applications (domaine compilation)

| Usage | Apport |
|---|---|
| **Phase-ordering** | scorer des séquences d'optimisation sans les exécuter |
| **Auto-tuning de compilateur** | choisir `-Ox` / passes adaptées à *ce* code |
| **Pré-filtrage** | écarter les optimisations sans gain prédit avant de tester en réel |
| **Représentation réutilisable** | embedding « universel » de code machine pour d'autres tâches en aval |

Au-delà : similarité de binaires, détection de clones/vulnérabilités, code search
sémantique — toute tâche qui bénéficie d'un embedding sémantique de programme.

## 6. L'argument qui claque

> Aujourd'hui, savoir si une optimisation accélère un programme exige de le
> **compiler et l'exécuter**. Nous prédisons son effet **dans l'espace latent,
> sans exécution** — et on le démontre : sur le saut d'optimisation principal
> O0→O1, notre predictor reconstruit l'embedding optimisé (cos 0.79 → 0.92) sur du
> code jamais vu.

## 7. Honnêteté scientifique (limites)

- **Le predictor brille sur O0→O1** (cos 0.79→0.92), mais **O1≈O2≈O3** : on a mesuré
  (800 programmes) que sur AnghaBench, **l'IR de -O2 et -O3 est identique à 100 %**.
  Clang sature dès -O2.
- **La vraie limite = le corpus**, pas la méthode : AnghaBench = fonctions *isolées
  et courtes*, donc -O3 (inlining inter-procédural, vectorisation) n'a rien à
  optimiser de plus que -O2. Le code n'est pas assez complexe pour distinguer les
  niveaux élevés. (Détail : `docs/limitation_non_bijective.md`.)
- L'IR est **déterministe** : le JEPA n'apprend pas de l'aléatoire, il apprend une
  *représentation* (comme BERT sur du texte). La prédiction non-triviale, c'est le
  predictor (deviner emb(O1) sans compiler).
- Pas de **temps d'exécution** réel (corpus non exécutable) → on prédit l'effet
  *structurel* de l'optimisation, pas le speedup en secondes.
- **Prochaine étape** : corpus de programmes complets et exécutables (ExeBench) où
  -O3 fait une vraie différence et où l'on peut relier les embeddings à des temps
  réels.