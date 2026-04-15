import numpy as np
import os
import math
import pickle

def lireMail(fichier, dictionnaire):
	""" 
	Lire un fichier et retourner un vecteur de booléens en fonctions du dictionnaire
	"""
	f = open(fichier, "r",encoding="ascii", errors="surrogateescape")
	mots = f.read().lower().split(" ")
	
	x = [False] * len(dictionnaire) 

	# on met True aux mots du dictionnaire qui sont présents dans le mail
	for i in range(len(dictionnaire)):
		if dictionnaire[i] in mots:
			x[i] = True
	
	f.close()
	return x

def charge_dico(fichier):
	f = open(fichier, "r")
	mots = f.read().split("\n")
	f.close()

	# on ne garde que les mots dont la longueur est >= 3 et on les met en minuscules
	mots_filtres = []
	for mot in mots:
		if len(mot) >= 3:
			mots_filtres.append(mot.lower())
	mots=mots_filtres

	print("Chargé " + str(len(mots)) + " mots dans le dictionnaire")
	return mots

def apprendBinomial(dossier, fichiers, dictionnaire, lissage=True):
	"""
	Fonction d'apprentissage d'une loi binomiale a partir des fichiers d'un dossier
	Retourne un vecteur b de paramètres 
	Conforme au sujet : on parcourt la base mail par mail (un fichier ouvert à la fois).
	"""
	m = len(fichiers)
	# compteur du nombre de mails contenant chaque mot j
	n_j = [0] * len(dictionnaire)

	for idx, nom_f in enumerate(fichiers):
		chemin = os.path.join(dossier, nom_f)
		x = lireMail(chemin, dictionnaire)   # liste de booléens
		# incrémenter pour chaque mot présent
		for j, present in enumerate(x):
			if present:
				n_j[j] += 1

	# lissage de Laplace epsilon = 1 
	if lissage == True:
		e = 1
		# bj = (n_j + e) / (m + 2*e) // formule du sujet
		b = [ (n + e) / (m + 2.0*e) for n in n_j ]
	else:
		#sans lissage, e=0
		b = [ n / m for n in n_j ] 
	return b



def prediction(x, Pspam, Pham, bspam, bham):
	"""
		Prédit si un mail représenté par un vecteur booléen x est un spam
		à partir du modèle de paramètres Pspam, Pham, bspam, bham.
		Retourne True ou False.
		
	"""
	# On essaie de calculer P(spam|x) et P(ham|x)
	#conversion en log pour éviter les problèmes numériques
	logPSpamX = np.log(Pspam)
	logPHamX = np.log(Pham)

	for i in range(len(x)):
		if x[i]: 
			logPSpamX += np.log(bspam[i])
			logPHamX += np.log(bham[i])
		else: 
			logPSpamX += np.log(1 - bspam[i])
			logPHamX += np.log(1 - bham[i])

	# On quitte log pour calculer les vraies proba
	PSpam_exp = np.exp(logPSpamX)
	PHam_exp = np.exp(logPHamX)

	# calcul de P(x)
	px = PSpam_exp + PHam_exp

	#Calcul de  P(spam|x) et P(ham|x) Formule de Bayes
	Pspam_x = PSpam_exp /px
	Pham_x = PHam_exp / px

	isSpam = Pspam_x > Pham_x

	return isSpam, Pspam_x, Pham_x
	 
	
def test(dossier, isSpam, Pspam, Pham, bspam, bham):
	"""
		Test le classifieur de paramètres Pspam, Pham, bspam, bham 
		sur tous les fichiers d'un dossier étiquetés 
		comme SPAM si isSpam et HAM sinon
		
		Retourne le taux d'erreur 
	"""
	fichiers = os.listdir(dossier)
	nb_erreurs = 0
	label_reel = "SPAM" if isSpam else "HAM"
	idx = 0  # compteur pour numéroter les mails

	for fichier in fichiers:
		print("Mail " + dossier+"/"+fichier)

		# lecture et conversion du mail en vecteur booléen
		x = lireMail(dossier+"/"+fichier, dictionnaire)
		# prédiction et récupération des probabilités a posteriori
		pred_isSpam, Pspam_x, Pham_x = prediction(x, Pspam, Pham, bspam, bham)
		label_predit = "SPAM" if pred_isSpam else "HAM"

		# affichage des probabilités a posteriori
		print(f"{label_reel} numéro {idx} : P(Y=SPAM | X=x) = {Pspam_x:.6e}, P(Y=HAM | X=x) = {Pham_x:.6e}")

		# détection des erreurs de classification
		if pred_isSpam != isSpam:
			nb_erreurs += 1
			print(f"   => identifié comme un {label_predit} *** erreur ***")
		else:
			print(f"   => identifié comme un {label_predit}")

		idx += 1

	return (nb_erreurs / len(fichiers)) * 100  

#Améliorations
def creerClassifieur(Pspam, Pham, bspam, bham, mSpam, mHam, dictionnaire):
	"""
	Encapsule tous les paramètres du classifieur dans un dictionnaire
	"""
	classifieur = {}
	classifieur["Pspam"] = Pspam
	classifieur["Pham"] = Pham
	classifieur["bspam"] = bspam
	classifieur["bham"] = bham
	classifieur["mSpam"] = mSpam
	classifieur["mHam"] = mHam # nb total de hams
	classifieur["dictionnaire"] = dictionnaire # on garde le dico dans le classifieur
	return classifieur

def testClassifieur(classifieur, dossier, isSpam):
	"""
	Version améliorée de tes qui prend directement l'objet classifieur
	plutôt que la liste de paramètres séparés.
	Retourne le taux d'erreur
	"""
	Pspam = classifieur["Pspam"]
	Pham = classifieur["Pham"]
	bspam = classifieur["bspam"]
	bham = classifieur["bham"]
	dictionnaire = classifieur["dictionnaire"]

	fichiers = os.listdir(dossier)
	nb_erreurs = 0
	label_reel = "SPAM" if isSpam else "HAM"

	for idx, fichier in enumerate(fichiers):
		print("Mail " + dossier+"/"+fichier)

		# lecture et conversion du mail en vecteur booléen
		x = lireMail(dossier+"/"+fichier, dictionnaire)
		# prédiction et récupération des probabilités a posteriori
		pred_isSpam, Pspam_x, Pham_x = prediction(x, Pspam, Pham, bspam, bham)
		label_predit = "SPAM" if pred_isSpam else "HAM"

		# affichage des probabilités a posteriori
		print(f"{label_reel} numéro {idx} : P(Y=SPAM | X=x) = {Pspam_x:.6e}, P(Y=HAM | X=x) = {Pham_x:.6e}")

		# détection des erreurs de classification
		if pred_isSpam != isSpam:
			nb_erreurs += 1
			print(f"   => identifié comme un {label_predit} *** erreur ***")
		else:
			print(f"   => identifié comme un {label_predit}")

	return (nb_erreurs / len(fichiers)) * 100

#enregistrer le classifieur
def enregistrerClassifieur(classifieur, fichier):
	"""
	Sauvegarde le classifieur dans un fichier
	"""
	with open(fichier, "wb") as f:
		pickle.dump(classifieur, f)
	print(f"Classifieur enregistré dans {fichier}")

#charger le classifieur
def chargerClassifieur(fichier):
	"""
	Charge le classifieur à partir d'un fichier
	"""
	with open(fichier, "rb") as f:
		classifieur = pickle.load(f)
	print(f"Classifieur chargé depuis {fichier}")
	return classifieur

############ programme principal ############

dossier_spams = "spam/baseapp/spam"	# à vérifier
dossier_hams = "spam/baseapp/ham"

fichiersspams = os.listdir(dossier_spams)
fichiershams = os.listdir(dossier_hams)

mSpam = len(fichiersspams)
mHam = len(fichiershams)

# Chargement du dictionnaire:
dictionnaire = charge_dico("spam/dictionnaire1000en.txt")
print(dictionnaire)

# Apprentissage des bspam et bham:
print("apprentissage de bspam...")
bspam = apprendBinomial(dossier_spams, fichiersspams, dictionnaire)
print("apprentissage de bham...")
bham = apprendBinomial(dossier_hams, fichiershams, dictionnaire)

# Sans le lissage
print("Apprentissage de bspam sans lissage de Laplace...")
bspam_sansLissage = apprendBinomial(dossier_spams, fichiersspams, dictionnaire, lissage=False)
print("apprentissage de bham sans lissage de Laplace...")
bham_sansLissage = apprendBinomial(dossier_hams, fichiershams, dictionnaire, lissage=False)


# Calcul des probabilités a priori Pspam et Pham:
Pspam = mSpam / (mSpam + mHam)
Pham = mHam / (mSpam + mHam)

# Création du classifieur et sauvegarde dans un fichier:
classifieur = creerClassifieur(Pspam, Pham, bspam, bham, mSpam, mHam, dictionnaire)
enregistrerClassifieur(classifieur, "classifieur.pkl")
classifieur = chargerClassifieur("classifieur.pkl")

# Calcul des erreurs avec la fonction test():

erreur_spam = test("spam/basetest/spam", True, Pspam, Pham, bspam, bham)
erreur_ham = test("spam/basetest/ham", False, Pspam, Pham, bspam, bham)
erreur_spam_sansLissage = test("spam/basetest/spam", True, Pspam, Pham, bspam_sansLissage, bham_sansLissage)
erreur_ham_sansLissage = test("spam/basetest/ham", False, Pspam, Pham, bspam_sansLissage, bham_sansLissage)

# Calcul des erreurs avec la fonction testClassifieur():

erreur_spam_class = testClassifieur(classifieur, "spam/basetest/spam", True)
erreur_ham_class = testClassifieur(classifieur, "spam/basetest/ham", False)

mSpamTest = len(os.listdir("spam/basetest/spam"))
mHamTest = len(os.listdir("spam/basetest/ham"))
total_mails = mSpamTest + mHamTest
erreur_globale = (erreur_spam * mSpamTest + erreur_ham * mHamTest) / total_mails
erreur_globale_class = (erreur_spam_class * mSpamTest + erreur_ham_class * mHamTest) / total_mails

# Avec le lissage
print(f"\nErreur de test sur {mSpamTest} SPAM : {erreur_spam:.0f} %")
print(f"Erreur de test sur {mHamTest} HAM  : {erreur_ham:.0f} %")
# Sans le lissage
print(f"\nErreur de test sans lissage sur {mSpamTest} SPAM : {erreur_spam_sansLissage:.0f} %")
print(f"Erreur de test sans lissage sur {mHamTest} HAM  : {erreur_ham_sansLissage:.0f} %")

print(f"Erreur de test globale sur {total_mails} mails : {erreur_globale:.0f} %")

print(f"\nErreur de test avec classifieur sur {mSpamTest} SPAM : {erreur_spam_class:.0f} %")
print(f"Erreur de test avec classifieur sur {mHamTest} HAM  : {erreur_ham_class:.0f} %")
print(f"Erreur de test globale avec classifieur sur {total_mails} mails : {erreur_globale_class:.0f} %")
