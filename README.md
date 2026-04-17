# Projet IA — Filtre anti-spam par classifieur naïf de Bayes

**M1 Informatique — Université de Lorraine, FST Nancy**  
**Auteurs :** Kaba SAKHO & Afiwa Aimée Kodjo

## Description

Filtre anti-spam basé sur le classifieur naïf de Bayes avec modèle binomial de textes.
Le programme apprend à distinguer les spams des hams (mails légitimes) à partir d'une base d'apprentissage, puis teste ses prédictions sur une base de test.

## Structure du projet


```
projetAi-filtr-anti-spam/
|-- spam/
│   |-- baseapp/
│   │   |-- ham/                  (2500 hams d'apprentissage)
│   │   |-- spam/                 (500 spams d'apprentissage)
│   |--basetest/
│   │   |-- ham/                  (500 hams de test)
│   │   |-- spam/                 (500 spams de test)
│   |-- dictionnaire1000en.txt    (1000 mots anglais courants)
|-- tpspam.py                     (programme principal)
|-- rapport_ia_final.docx         (rapport du projet)
|-- README.md
|--classifieur.pkl et classifieur_maj.pkl

```

## Prérequis

- Python 3
- NumPy (`pip install numpy`)

## Lancement

```bash
python3 tpspam.py
```

Le programme effectue automatiquement :
1. Chargement du dictionnaire (950 mots après filtrage)
2. Apprentissage sur 500 spams et 2500 hams (avec lissage ε=1)
3. Test sur 500 spams et 500 hams de test
4. Sauvegarde du classifieur dans `classifieur.pkl`
5. Test de l'apprentissage en ligne (+10 spams)

## Résultats

| Configuration | Err. SPAM | Err. HAM | Err. globale |
|---|---|---|---|
| Sans lissage | 36 % | 1 % | 19 % |
| Avec lissage (ε=1) | 27 % | 1 % | 14 % |

## Fonctions principales

- `charge_dico(fichier)` -  charge le dictionnaire (mots ≥ 3 lettres, en minuscules)
- `lireMail(fichier, dictionnaire)` -  encode un mail en vecteur de booléens
- `lireMailAmeliore(fichier, dictionnaire)` - version optimisée (regex + set)
- `apprendBinomial(dossier, fichiers, dictionnaire, lissage=True)` -  apprend les paramètres du modèle
- `prediction(x, Pspam, Pham, bspam, bham)` -  prédit SPAM ou HAM avec probas a posteriori
- `test(dossier, isSpam, Pspam, Pham, bspam, bham)` -  teste le classifieur sur un dossier
- `creerClassifieur(...)` - encapsule les paramètres dans un dictionnaire
- `testClassifieur(classifieur, dossier, isSpam)` -  teste avec l'objet classifieur
- `enregistrerClassifieur(classifieur, fichier)` -- sauvegarde avec pickle
- `chargerClassifieur(fichier)` -  charge depuis pickle
- `mettreAJour(classifieur, chemin_mail, est_spam)` - apprentissage en ligne
