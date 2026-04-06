import numpy as np
import os
import math

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

def apprendBinomial(dossier, fichiers, dictionnaire):
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
	e = 1
	# bj = (n_j + e) / (m + 2*e) // formule du sujet
	b = [ (n + e) / (m + 2.0*e) for n in n_j ]
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
	for fichier in fichiers:
		print("Mail " + dossier+"/"+fichier)		

		
		# à compléter...

	return 0  # à modifier...


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


# Calcul des probabilités a priori Pspam et Pham:
Pspam = mSpam / (mSpam + mHam)
Pham = mHam / (mSpam + mHam)


# Calcul des erreurs avec la fonction test():


