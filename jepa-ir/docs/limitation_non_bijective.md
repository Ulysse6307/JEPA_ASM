# O2 vs O3 sur AnghaBench : clang sature dès -O2

> ⚠️ Ce document a été **corrigé**. Une première version affirmait une « perte
> d'information de notre graphe (36 %) » — c'était une **erreur de mesure**
> (échantillons O2/O3 mal alignés). Le résultat vérifié sur 800 programmes est
> tout autre, et plus net.

## 1. Le constat de départ

Le predictor par transition (cf. `predictor_per_step_analysis.md`) montre que les
embeddings de O1, O2, O3 sont quasi-confondus. On a cherché pourquoi, en
distinguant deux causes possibles :

- **H1** — notre représentation `IR → graphe` perd de l'info (non-bijective).
- **H2** — l'IR lui-même est déjà identique entre niveaux (clang sature).

## 2. Test vérifié (800 programmes, corpus complet, pool predictor)

Pour le même programme compilé à -O2 et -O3, on compare l'**IR textuel** (avant
notre graphe) et la **signature du graphe** :

| Mesure (O2 vs O3) | Résultat |
|---|---|
| IR identique | **100 %** |
| graphe identique | **100 %** |
| IR différent mais graphe identique (= perte d'info) | **0 %** |

*(Reproduit sur 250 puis 800 programmes — stable.)*

## 3. Conclusion : c'est H2, pas H1

**Sur AnghaBench, clang atteint sa limite d'optimisation dès -O2.** Pour 100 % des
fonctions testées, l'IR de -O2 et de -O3 est *littéralement identique* — clang ne
trouve **rien de plus** à optimiser en -O3.

→ Donc O2 et O3 ont le même embedding **légitimement** : ils sont le même code.
Ce **n'est PAS** une perte d'information de notre représentation : notre graphe est
identique parce que l'IR d'entrée est identique. (La version précédente du document
se trompait sur ce point.)

## 4. Pourquoi clang sature si tôt sur ce corpus

AnghaBench = **fonctions isolées**, extraites de leur contexte (headers/types
synthétisés, pas de `main`, pas d'appelants réels). Or les optimisations que -O3
ajoute par rapport à -O2 sont surtout :

- **inter-procédurales** (inlining agressif entre fonctions) — sans contexte
  d'appel, rien à inliner ;
- **vectorisation de boucles longues** — les fonctions sont courtes ;
- **unrolling agressif** — peu de boucles à dérouler.

Sur des fonctions courtes et isolées, **-O2 capture déjà tout**. C'est cohérent
avec la littérature : -O3 ne se distingue de -O2 que sur du code plus gros et plus
« connecté ».

## 5. Ce que ça implique pour le projet

- Le predictor « chaîne O0→O1→O2→O3 » n'a de **transition non-triviale que
  O0→O1** ; O1→O2 et O2→O3 sont (quasi) l'identité parce que le code ne change pas.
- Le résultat fort et honnête : sur O0→O1 (le seul vrai saut), le predictor
  reconstruit l'embedding optimisé avec **cos 0.79 → 0.92**.
- Pour observer O2 vs O3, il faudrait un **corpus exécutable et plus gros**
  (ExeBench, programmes complets) où -O3 fait réellement une différence.

## 5 bis. La vraie limite : le CORPUS, pas la méthode

Le fond du problème n'est ni l'encodeur ni la représentation : **AnghaBench est
trop simple pour distinguer les niveaux d'optimisation élevés.**

Hiérarchie de saturation mesurée (graphes bruts, % de programmes où le niveau
change quelque chose) :

| Transition | graphes différents | ce que clang fait |
|---|---|---|
| O0 → O1 | ~89 % (gros saut) | SSA, mem2reg, nettoyage pile |
| O1 → O2 | ~16 % (petit) | quelques optimisations locales |
| O2 → O3 | **0 %** (rien) | inlining inter-proc. / vectorisation → *rien à mordre* |

Les fonctions AnghaBench sont **isolées et courtes** (~22 nœuds médians). Or -O3
cible précisément ce qu'elles n'ont pas : appels à inliner, longues boucles à
vectoriser, déroulage. **Le code n'est pas assez complexe pour que -O3 serve.**

C'est un **résultat en soi**, pas un échec : on a quantifié le rendement décroissant
de l'optimisation sur du code isolé. Pour observer une vraie différence O2/O3, il
faudrait un corpus de **programmes complets et plus gros** (p. ex. ExeBench), où
-O3 a matière à travailler — ce serait l'extension naturelle du projet.

## 6. Note de méthode (transparence)

L'erreur initiale (36 %) venait d'une comparaison sur des listes de fichiers mal
alignées entre O2 et O3 (échantillon `sample_100k` indexé). Le test corrigé
recompile le **même fichier** aux deux niveaux et compare directement — d'où le
0 % de perte d'info. Leçon : toujours vérifier l'alignement des paires avant de
conclure à une limite de méthode.