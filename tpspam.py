import numpy as np
import os
import math
import pickle
import re # lireMailAmeliore

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
	
def lireMailAmeliore(fichier, dictionnaire):
	"""
	Version améliorée de lireMail :
	1. split par regex sur tout caractère non alphabétique (gère ponctuation, \n, \t)
	2. utilisation d'un set pour le lookup en O(1) au lieu de O(n)
	"""
	f = open(fichier, "r", encoding="ascii", errors="surrogateescape")
	mots = set(re.split(r"[^a-z]+", f.read().lower()))
	f.close()

	x = [False] * len(dictionnaire)
	for i in range(len(dictionnaire)):
		if dictionnaire[i] in mots:
			x[i] = True

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
	"""
	m = len(fichiers)
	cpt = np.zeros(len(dictionnaire))

	for nom_f in fichiers:
		chemin = os.path.join(dossier, nom_f)
		x = lireMail(chemin, dictionnaire)
		cpt += x 

	if lissage:
		e = 1
		b = (cpt + e) / (m + 2.0 * e)
	else:
		b = cpt / m

	return b



def prediction(x, Pspam, Pham, bspam, bham):
	"""
		Prédit si un mail représenté par un vecteur booléen x est un spam
		à partir du modèle de paramètres Pspam, Pham, bspam, bham.
		Retourne True ou False.
	"""
	x = np.array(x, dtype=bool)
	bspam = np.array(bspam)
	bham = np.array(bham)

	# log-vraisemblances
	logPSpamX = np.log(Pspam) + np.sum(np.log(bspam[x])) + np.sum(np.log(1 - bspam[~x]))
	logPHamX  = np.log(Pham)  + np.sum(np.log(bham[x]))  + np.sum(np.log(1 - bham[~x]))

	# probabilités a posteriori
	PSpam_exp = np.exp(logPSpamX)
	PHam_exp = np.exp(logPHamX)
	px = PSpam_exp + PHam_exp
	Pspam_x = PSpam_exp / px
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
	idx = 0 

	for fichier in fichiers:
		print("Mail " + dossier+"/"+fichier)
		x = lireMail(dossier+"/"+fichier, dictionnaire)
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
	classifieur["mHam"] = mHam
	classifieur["dictionnaire"] = dictionnaire 
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

		print(f"{label_reel} numéro {idx} : P(Y=SPAM | X=x) = {Pspam_x:.6e}, P(Y=HAM | X=x) = {Pham_x:.6e}")

		if pred_isSpam != isSpam:
			nb_erreurs += 1
			print(f"   => identifié comme un {label_predit} *** erreur ***")
		else:
			print(f"   => identifié comme un {label_predit}")

	return (nb_erreurs / len(fichiers)) * 100

def enregistrerClassifieur(classifieur, fichier):
	"""
	Sauvegarde le classifieur dans un fichier
	"""
	with open(fichier, "wb") as f:
		pickle.dump(classifieur, f)
	print(f"Classifieur enregistré dans {fichier}")

def chargerClassifieur(fichier):
	"""
	Charge le classifieur à partir d'un fichier
	"""
	with open(fichier, "rb") as f:
		classifieur = pickle.load(f)
	print(f"Classifieur chargé depuis {fichier}")
	return classifieur


#ks
def mettreAJour(classifieur, chemin_mail, est_spam):
	"""
	Apprentissage en ligne : met à jour le classifieur avec un seul nouveau mail.
	Formule du sujet (page 3) :
	    b_j(m+1) = (n_j + x_j + eps) / (m + 1 + 2*eps)
	"""
	dico = classifieur["dictionnaire"]
	x = lireMail(chemin_mail, dico)
	e = 1  

	if est_spam:
		# retrouver n_j à partir de b_j
		m = classifieur["mSpam"]
		ancien_b = classifieur["bspam"]
		nouveau_b = []
		for j in range(len(dico)):
			n_j = ancien_b[j] * (m + 2*e) - e
			n_j_new = n_j + (1 if x[j] else 0)
			nouveau_b.append((n_j_new + e) / (m + 1 + 2*e))
		classifieur["bspam"] = nouveau_b
		classifieur["mSpam"] = m + 1
	else:
		m = classifieur["mHam"]
		ancien_b = classifieur["bham"]
		nouveau_b = []
		for j in range(len(dico)):
			n_j = ancien_b[j] * (m + 2*e) - e
			n_j_new = n_j + (1 if x[j] else 0)
			nouveau_b.append((n_j_new + e) / (m + 1 + 2*e))
		classifieur["bham"] = nouveau_b
		classifieur["mHam"] = m + 1

	# mise à jour des probabilités a priori
	total = classifieur["mSpam"] + classifieur["mHam"]
	classifieur["Pspam"] = classifieur["mSpam"] / total
	classifieur["Pham"] = classifieur["mHam"] / total

	return classifieur
	
############ programme principal ############

dossier_spams = "spam/baseapp/spam"	
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


#ks
# apprentissage en ligne
print("\n APPRENTISSAGE EN LIGNE")
print(f"mSpam avant : {classifieur['mSpam']}")
# on ajoute 10 spams au classifieur
for i in range(10):
	fichiers_test = os.listdir("spam/baseapp/spam")
	classifieur = mettreAJour(classifieur, "spam/baseapp/spam/" + fichiers_test[i], True)
print(f"mSpam après ajout de 10 spams : {classifieur['mSpam']}")
print(f"Pspam mis à jour : {classifieur['Pspam']:.4f}")

# on re-teste après mise à jour
erreur_spam_maj = testClassifieur(classifieur, "spam/basetest/spam", True)
erreur_ham_maj = testClassifieur(classifieur, "spam/basetest/ham", False)
erreur_globale_maj = (erreur_spam_maj * mSpamTest + erreur_ham_maj * mHamTest) / total_mails
print(f"\nAprès apprentissage en ligne (+10 spams) :")
print(f"Erreur SPAM : {erreur_spam_maj:.0f} %")
print(f"Erreur HAM  : {erreur_ham_maj:.0f} %")
print(f"Erreur globale : {erreur_globale_maj:.0f} %")

# sauvegarde du classifieur mis à jour
enregistrerClassifieur(classifieur, "classifieur_maj.pkl")



