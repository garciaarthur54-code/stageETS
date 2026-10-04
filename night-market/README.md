# Night Market Tycoon 🏮

Prototype jouable d'un tycoon « idle arcade » mobile : tu gères un marché de nuit qui grandit, de la ruelle au festival.

## Jouer

Ouvre `index.html` dans un navigateur (mobile ou ordinateur). Le jeu tient dans un seul fichier, sans dépendance ni build.

- **Mobile** : glisse le doigt n'importe où pour te déplacer (joystick virtuel).
- **Ordinateur** : flèches, ZQSD ou WASD.

## Boucle de jeu

1. Les stands cuisinent tout seuls. Passe devant l'assiette pour **ramasser** les plats.
2. Approche-toi du **comptoir** pour y poser les plats. Les clients se servent.
3. Va chercher les **pièces** (💰 à droite du comptoir).
4. Tiens-toi immobile sur une **case en pointillés** pour y verser ton argent et débloquer :
   un nouveau stand, un **vendeur** (il fait les allers-retours à ta place) ou un **caissier** (il encaisse pour toi).
5. Utilise **Améliorer** pour monter le niveau des stands (prix, vitesse, stock), tes baskets et ton plateau.
6. Une fois les 4 stands ouverts, **Déménage** dans une nouvelle ville : tu repars de zéro, mais avec des gains ×2 permanents.

Bonus : environ 4 % des clients sont **VIP** 👑 et paient ×5. Si tu as des vendeurs, le marché continue de rapporter **hors ligne** pendant 3 h maximum.
La partie est sauvegardée automatiquement dans le navigateur (localStorage).

## Pistes pour la suite

- Recettes fusion (🍜 + 🌮) et livre de recettes à compléter
- Festivals hebdomadaires avec un stand exclusif
- Visite du marché des amis et pourboires
- Pubs récompensées (×2 hors ligne, client VIP), pass saisonnier, cosmétiques
- Portage vers un moteur mobile (Phaser + Capacitor, Unity ou Godot) pour publier sur les stores
