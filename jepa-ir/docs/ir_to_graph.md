# 🔬 De l'IR au graphe : comment on construit la représentation

> Comment on transforme du code en notre graphe **3-relations sans perte**, qui est
> l'entrée de l'encodeur GNN. Basé sur le code réel : [`graph/builder.py`](../src/jepa_ir/graph/builder.py).

## Vue d'ensemble

```
  code C/C++  ──clang──▶  LLVM IR (texte)  ──builder──▶  ProgramGraph (3 relations)
```

Le builder prend le **texte** de l'IR LLVM et en extrait un graphe où :

- **chaque nœud = une instruction LLVM**
- **chaque arête = une relation typée** parmi les 3 : `control`, `data`, `memory`

> 💡 **Pourquoi parser le texte ?** llvmlite n'expose pas le graphe d'opérandes en
> Python (ça vit dans l'API C++). Le texte IR, lui, est stable et complet — on le lit
> directement. llvmlite sert juste à **vérifier** que l'IR est valide
> (`parse_assembly` + `verify`).

---

## Exemple fil rouge

Prenons cette fonction C :

```c
int sum_array(const int *a, int n) {
    int sum = 0;
    for (int i = 0; i < n; i++)
        sum += a[i];
    return sum;
}
```

`clang -O1 -emit-llvm` la transforme en IR (extrait simplifié) :

```llvm
define i32 @sum_array(ptr %a, i32 %n) {
entry:
  %cmp4 = icmp sgt i32 %n, 0
  br i1 %cmp4, label %for.body, label %for.end      ; ← branchement

for.body:
  %i    = phi i64 [ 0, %entry ], [ %i.next, %for.body ]
  %sum  = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %ptr  = getelementptr i32, ptr %a, i64 %i
  %0    = load i32, ptr %ptr                          ; ← lecture mémoire
  %add  = add i32 %0, %sum                            ; ← %0 et %sum utilisés ici
  %i.next = add i64 %i, 1
  br i1 %cond, label %for.end, label %for.body

for.end:
  %res = phi i32 [ 0, %entry ], [ %add, %for.body ]
  ret i32 %res
}
```

Chaque ligne deviendra **un nœud**. Voyons les 4 étapes du builder.

---

## Étape 1 — Découper en fonctions et en nœuds

Le builder repère chaque `define ... { ... }` (`_split_functions`), puis parcourt les
lignes. Pour chaque instruction, il extrait via regex :

| Champ | Comment | Exemple (`%add = add i32 %0, %sum`) |
|---|---|---|
| **opcode** | 1er mot après les modificateurs (`tail`, `nsw`, `atomic`…) | `add` |
| **résultat (def)** | la valeur `%x =` à gauche | `%add` |
| **opérandes** | toutes les valeurs `%`/`@` lues | `[%0, %sum]` |
| **flags** | terminator ? mémoire ? produit une valeur ? | — |

```
[0] entry    icmp     def=%cmp4    ops=[%n]
[1] entry    br       def=None     ops=[%cmp4, ...]      ← terminator
[2] for.body phi      def=%i       ops=[...]
[3] for.body phi      def=%sum     ops=[...]
[4] for.body getelem  def=%ptr     ops=[%a, %i]
[5] for.body load     def=%0       ops=[%ptr]            ← mémoire
[6] for.body add      def=%add     ops=[%0, %sum]
...
```

Chaque nœud retient aussi son **bloc de base** (`entry`, `for.body`…) — crucial pour
les arêtes de contrôle et de mémoire.

---

## Étape 2 — Arêtes **DATA** (flot de données) 🔵

> **Règle :** le nœud qui *définit* une valeur `%x` → tout nœud qui *utilise* `%x`
> comme opérande.

On construit d'abord une table `def_site` (*quelle instruction produit `%x` ?*), puis
pour chaque opérande de chaque nœud, on relie sa source à son utilisateur :

```python
for node in nodes:
    for op in node.operands:
        src = def_site.get(op)        # qui définit cet opérande ?
        if src is not None and src != node.idx:
            add_edge("data", src, node.idx)
```

Sur l'exemple : `%0 = load ...` puis `%add = add %0, %sum` → arête **data**
`load(5) → add(6)`. C'est le flot SSA : la donnée circule de sa production à sa
consommation. Les `phi` créent les cycles caractéristiques des boucles.

---

## Étape 3 — Arêtes **CONTROL** (flot de contrôle) 🔴

> **Règle :** le *terminator* d'un bloc → la *première instruction* de chaque bloc
> successeur.

Les terminators (`br`, `switch`, `ret`, `indirectbr`…) nomment leurs blocs cibles via
`label %xxx`. On les résout :

```python
# br i1 %cmp4, label %for.body, label %for.end
targets = ["for.body", "for.end"]
for lab in targets:
    add_edge("control", terminator_idx, first_node_of[lab])
```

Sur l'exemple : `br(1)` dans `entry` → `phi(2)` (1ʳᵉ instr de `for.body`) **et** → la
1ʳᵉ instr de `for.end`. C'est le CFG, mais **porté au niveau de l'instruction** (pas du
bloc), ce qui le rend homogène avec les deux autres relations.

---

## Étape 4 — Arêtes **MEMORY** (ordre des effets de bord) 🟢

> **C'est NOTRE relation distinctive** — celle qui manque à un simple CFG/AST
> (et à ProGraML, qui fait control+data+call mais pas memory).

> **Règle :** on chaîne dans l'**ordre du programme** les instructions qui touchent la
> mémoire (`load`, `store`, `call`, `invoke`, `atomicrmw`, `cmpxchg`, `fence`).

```python
# dans chaque bloc : ordre textuel des instructions mémoire
mem_nodes = [n for n in block if n.is_memory_op]
for a, b in zip(mem_nodes, mem_nodes[1:]):
    add_edge("memory", a, b)          # a précède b

# entre blocs : dernière instr mémoire de B → 1ère du successeur (suit le CFG)
```

Ça encode **la séquence des effets observables** : si deux `store` s'enchaînent, leur
ordre compte (`p[0]=1; p[0]=2` ≠ `p[0]=2; p[0]=1`). Un CFG classique perd cette info ;
nous la gardons.

> ⚠️ **Honnêteté sur la précision :** c'est un **ordre de programme conservateur**, pas
> une analyse d'alias précise. On ne résout pas si deux pointeurs visent la même case —
> on encode juste *l'ordre* dans lequel les effets surviennent. C'est exactement ce que
> demande la spec (« ordre des effets de bord »), qui parle d'ordre, pas de dépendance
> alias-précise.

---

## Le résultat : un multigraphe à 3 relations

Pour `sum_array`, le builder produit (chiffres réels du run) :

```
ProgramGraph 'sum_array'  nodes=14
  ├─ control : 5 arêtes   🔴  (branches du for-loop)
  ├─ data    : 13 arêtes  🔵  (def→use SSA, cycles des phi)
  └─ memory  : 1+ arêtes  🟢  (ordre des load/store)
```

**Le même ensemble de nœuds (instructions), trois "couches" d'arêtes superposées :**

```
        ┌─────────────────────────────────────┐
        │  NŒUDS  = instructions LLVM          │
        │  [icmp] [br] [phi] [load] [add] ...  │
        └─────────────────────────────────────┘
              ▲            ▲             ▲
        🔴 control     🔵 data      🟢 memory
        (qui saute    (qui produit  (quel ordre
         vers qui)     pour qui)     d'effets)
```

---

## Pourquoi 3 relations et pas 1 ?

| Représentation | Contrôle | Données | Mémoire |
|---|:---:|:---:|:---:|
| AST | ❌ | ❌ | ❌ |
| CFG simple | ✅ | ❌ | ❌ |
| ProGraML | ✅ | ✅ | ❌ |
| **Le nôtre** | ✅ | ✅ | ✅ |

Garder les **trois sans perte**, c'est ce qui permet à l'encodeur de « comprendre » un
programme dans sa totalité : sa structure (contrôle), ses dépendances (données), et
l'ordre observable de ses effets (mémoire). C'est l'hypothèse centrale du projet.

---

## Et après le graphe ?

```
ProgramGraph ──convert.py──▶ PyG Data ──masking──▶ 2 vues ──GNN──▶ embedding
                            (x_opcode,    (masquée +       (JEPA + VICReg)
                             3× edge_index) complète)
```

1. **`convert.py`** : le graphe devient des tenseurs PyTorch Geometric — un opcode
   embeddé + features structurels par nœud, et **un `edge_index` par relation**.
2. **`masking.py`** : on produit deux vues — une **masquée** (30 % des nœuds, par blocs)
   et la **complète**. Les nœuds masqués gardent leur place et leurs arêtes mais leur
   contenu est remplacé par un **mask token appris**.
3. **GNN** (`model/encoder.py`) : *message passing* séparé sur les 3 types d'arêtes,
   puis pooling → un vecteur (l'**embedding**, le livrable).
4. **Entraînement JEPA + VICReg** : on rapproche les embeddings des deux vues dans
   l'espace latent, avec VICReg pour interdire le collapse.

---

## Fichiers de référence

| Fichier | Rôle |
|---|---|
| [`graph/builder.py`](../src/jepa_ir/graph/builder.py) | IR texte → `ProgramGraph` (les 4 étapes ci-dessus) |
| [`graph/schema.py`](../src/jepa_ir/graph/schema.py) | la structure `ProgramGraph` / `Node` (classe maison) |
| [`ir/compile.py`](../src/jepa_ir/ir/compile.py) | C/C++ → LLVM IR via `clang -emit-llvm` |
| [`data/convert.py`](../src/jepa_ir/data/convert.py) | `ProgramGraph` → PyG `Data` |

Pour inspecter un graphe toi-même :

```bash
python scripts/inspect_graph.py examples/sum_array.c
```