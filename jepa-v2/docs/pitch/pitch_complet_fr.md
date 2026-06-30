# Pitch investisseur — version intégrale (FR)

> À dire d'un trait, ~4 min. Les **passages en gras** sont les points à marquer.
> Les chiffres exacts sont rappelés en bas — ne jamais les rater.

---

Bonjour. Nous construisons **le compilateur qui apprend.**

Pensez au code le plus exigeant qui existe — les moteurs de trading, les
simulations, les bases de données. Là, **chaque cycle de vitesse et chaque octet de
mémoire, c'est de l'argent**, et une chose n'est jamais négociable : le code doit
être **correct, jusqu'au métal**. Ces entreprises dépensent des fortunes pour la
performance — et pourtant elles en laissent énormément sur la table.

Pourquoi ? Parce qu'il y a une vérité qu'on oublie : **la performance ne se joue pas
dans votre code source. Elle se joue dans l'assembleur que produit le compilateur.**
C'est la couche qui décide de tout — et aujourd'hui, elle tourne en pilote
automatique.

Les compilateurs d'aujourd'hui reposent sur des **heuristiques écrites à la main** —
des règles qu'un expert a réglées il y a des années, appliquées dans un ordre figé,
avec des modèles de coût qui s'éloignent du matériel à chaque nouvelle génération de
puces. Comme ils doivent garantir la correction, ils jouent la sécurité et laissent de
la performance de côté. Et surtout : **ils n'apprennent jamais.** Chaque nouvelle
architecture, c'est encore du réglage manuel, à la main, par des experts. C'est un
tapis roulant, et il s'accélère.

Notre approche tient en deux paris. Le premier : **représenter le code comme un graphe
de flux de données** — qui dépend de quoi. C'est ça, la vraie structure qui pilote la
performance ; contrairement à un LLM, **nous ne travaillons jamais sur du texte**. Le
second : **apprendre sur ce graphe avec JEPA**, l'approche auto-supervisée de Yann
LeCun, ces « world models ». Elle commence à peine à être explorée ailleurs, en
biologie par exemple — et **à notre connaissance, personne ne l'a appliquée à
l'assembleur. Nous sommes les premiers.**

Concrètement : on prend l'assembleur, on le transforme en graphe de flux — contrôle,
données, appels — et on le passe dans un réseau de neurones sur graphes entraîné **à
partir de zéro, sans aucun label.** En sortie, un embedding **factorisé en deux** : ce
que le code fait, et son profil d'optimisation.

**Et ça marche.** Sur des programmes que le modèle n'a jamais vus, il **sépare
proprement les deux** : une moitié capture le sens du programme, indifférente au
niveau d'optimisation ; l'autre capture le profil d'optimisation, indifférente au
programme. Un écart de 0,89, décorrélé — vous le voyez sur la projection. **C'est
auto-supervisé, entraîné de zéro sur un seul GPU, c'est sur GitHub, ça se reproduit en
une commande, avec une suite de tests qui passe.** En 36 heures, nous sommes passés
de l'idée à un cœur fonctionnel et prouvé.

Et nous faisons de la vraie science, pas de la démo de façade. **Nous validons avant
d'entraîner** : nous avons mesuré honnêtement où le compilateur actuel sature, et nous
rapportons ce plafond au lieu de le cacher. Et nous creusons : un travail soigné sur
l'objectif a fait passer le modèle d'une poignée de dimensions à **72 sur 96 — sur les
mêmes données.** C'est la rigueur qu'exige la correction au niveau machine.

Pour aller chercher les clients, nous avons une recette éprouvée — **la stratégie
GitGuardian.** On scanne les dépôts open source, on repère le code qu'on peut
radicalement optimiser, et on contacte automatiquement le responsable avec une **démo
gratuite d'optimisation de son propre code.** Une preuve immédiate, irréfutable, qui
transforme l'open source en prospects qualifiés.

Et le modèle économique est superbe. Nos clients, ce sont **les équipes dont la facture
de calcul EST le métier** — trading haute fréquence, bases de données, HPC,
infrastructure ML. On facture à la valeur : **on prend une part du calcul qu'on fait
économiser.** Optimiser une boucle critique nous coûte quelques centimes ; le gain,
lui, se répète à chaque exécution, pour toujours. **5 % d'économie sur une facture de
calcul de 2 millions, c'est 100 000 économisés — on en garde 25 000 par an, par
charge de travail, à 90 % de marge.**

Cet encodeur, c'est la brique fondatrice de bien plus grand : **un compilateur
universel piloté par l'IA.** À terme, n'importe quel code source, dans n'importe quel
langage, compilé de façon optimale pour n'importe quel matériel — CPU, GPU, puces
spécialisées. Un world model du code qui devient plus intelligent à chaque programme
qu'il voit.

Et le moment, c'est maintenant. **Les world models JEPA arrivent dans de nouveaux
domaines, et l'assembleur est un terrain vierge.** Chaque nouvelle puce rend le réglage
manuel des compilateurs plus intenable. Et le calcul distribué bon marché permet à une
petite équipe comme la nôtre d'entraîner tout ça — pour une fraction du coût des clouds
classiques.

Alors voilà : en 36 heures, nous avons transformé un pari audacieux en **cœur prouvé et
reproductible.** Nous levons un **pre-seed pour passer de la factorisation au premier
vrai gain de vitesse sur du code de production**, et pour verrouiller le fossé de
données. **Faisons en sorte que le compilateur apprenne.** Merci — vos questions avec
plaisir.

---

## Réponses Q&A (à garder en poche)

- **« Est-ce que ça accélère vraiment le code aujourd'hui ? »**
  « Pas encore — et nous ne le prétendons pas. Nous avons prouvé la représentation. La
  prochaine étape, c'est un premier gain mesuré sur un vrai benchmark — et c'est
  exactement ce que ce tour finance. »
- **« Comment garantissez-vous la correction ? »**
  « Nous proposons, la chaîne de compilation prouve : chaque réécriture est validée par
  les vérifications d'équivalence du compilateur. On ne livre jamais une transformation
  non vérifiée. »
- **« Pourquoi Google ou un labo LLM ne vous écraseront pas ? »**
  « Parce qu'on apprend sur le graphe, pas sur le texte, et que le vrai fossé, c'est un
  corpus propriétaire de paires (graphe → gain de vitesse mesuré) qui se renforce avec
  chaque client. Ça ne se reproduit pas en un week-end. »

## Chiffres — à ne jamais rater
- Factorisation : écart z_sem **0,89** ; z_speed ignore le programme (**−0,91**).
- Capacité après réglage : **3 → 72** dimensions sur 96.
- Plafond mesuré honnêtement : O0→O1 change **100 %** des graphes ; O2≈O3 **identiques**.
- Échelle : **~8 000** fonctions · **un GPU B200** · **24 tests passent** · sur GitHub.
- Argent : **facture 2 M$ → 5 % → 100 k$ économisés → 25 k$/an par charge → ~90 % de marge.**

## Version 90 secondes
Enchaînez : stakes → problème → insight → preuve → modèle éco → ask.
(slides 2 → 4 → 5 → 7 → 10 → 16)
