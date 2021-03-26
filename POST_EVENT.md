# Les éléments de résolutions

Pour cette édition de Crack The Captcha, aucune méthode particulière n'était imposée pour résoudre automatiquement les
quelques 10 mille catchas inconnus. On aurait pu les faire à la main (avec l'aide de toute la famille), hacker le site
pour récupérer les solutions du jeu, utiliser des logiciels de reconnaissance de caractère (OCR). Mais
le challenge présenté ici était un cas parfait d'utilisation du machine learning et notamment des réseaux de neurones.
HOG + SVM, réseaux à couches denses (MLP), convolué (CNN), attention network, LSTM (RNN), les méthodes de résolution
ne manquent, alors essayons !

Pas si simple de créer from scratch et d'utiliser un réseau de neurones en moins de deux heures, commençons très
simplement par un réseau basique à une couche cachée dense. Mais pour cela il va nous falloir préparer les données.

L'ensemble des exemples de code présentés dans la suite sont des extraits du fichier développé pendant le live:
[demo.py](https://github.com/meritisgroup/challenge-crack-the-captcha-public/blob/master/demo.py)

## Entrainer un réseau

Pour l'entrainement, il va nous falloir travailler sur les entrées appelées `X` et les sorties attendues `Y` appelées
labels. Pour faciliter la tâche du réseau de neurone, je vais découper l'image pour extraire des lettres / chiffres.
La tâche d'apprentissage reviendra à classifier ces lettres. Pour résoudre mes captcha, je n'aurai qu'à découper les
lettres, prédire leurs labels et fusionner ces résultats pour récupérer le captcha complet.

Pour chaque image, je vais convertir chaque lettre en un vecteur de taille fixe :

```python
# Get the label contained in the filename
word, level, ext = filename.split(".")

# Load current image
image = cv2.imread(os.path.join(path, filename))

# Transform in gray 
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Separate the letters
for i, letter in enumerate(word):
    X.append(image[0:35, (i * 20):((i + 1) * 20)].flatten())
    Y.append(letter)

# Normalize X vector - get values between 0 and 1
X = np.array(X, dtype=np.float) / 255.0  # Normalization
```

Les labels ont aussi besoin d'être transformés / encodés. On ne peut pas dire au réseau que la résponse est un `R` par 
exemple, il faut lui donner un nombre ou un vecteur à atteindre pour cette lettre. Le plus courant pour la classification
à plusieurs classes (ici classes = lettres), est d'encoder nos différentes lettres en un vecteur binaire. Si par exemple,
je n'ai que les lettres A, B, C, D, j'ai quatre classes à identifier et je peux encoder comme cela : 

```pseudocode
A -> [1, 0, 0, 0]
B -> [0, 1, 0, 0]
C -> [0, 0, 1, 0]
D -> [0, 0, 0, 1]
```

On pourrait coder une fonction d'encodage à la main mais c'est un processus classique en machine learning alors autant
réutiliser les implementations existantes (présente dans scikit-learn par exemple)

```python
# Create an encoder based on my labels
lb = sklearn.preprocessing.LabelBinarizer().fit(Y)
# Transform all my labels in binary vectors
Y = lb.transform(Y)
# Just get the number of classes
nb_labels = len(lb.classes_)
```

Maintenant nous avant nos vecteurs images `X` et nos labels `Y` prêts à être utilisés par un réseau. il va nous falloir
décrire les différents couches (layers) et les différents hyperparamètres, une tâche délicate ! Restons simple avec 
seulement une couche cachée de 128 neurones. C'est parti !

```python
model = k.models.Sequential()
model.add(k.layers.Dense(128, input_dim=(20 * 35), activation='relu'))
model.add(k.layers.Dense(nb_labels, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(X, Y, validation_split=0.2, batch_size=128, epochs=20, verbose=1)
```

Quoi, c'est tout ?! Et oui, la description d'un réseau très simple avec `Keras` est un jeu d'enfant. C'est ici que vous
pouvez vous amuser à enrichir l'architecture du modèle. Plus de couches ? plus de neurones ? des couches de convolutions ?
c'est à vous de voir. Moi je reste sur ce petit réseau pour le moment. Il y a beaucoup à dire sur le choix des differents
hyper-paramètres, et de la structure du réseau, mais c'est par l'expérimentation que la plupart des modèles sont créés.

A ce stade, nous avant un modèle complet caché sous la variable `model`. Utilisons la pour cracker tous les captcha
inconnus !

## Inférence sur le jeu de test

La phase où l'on applique le réseau sur des données (sans l'entrainer) s'appelle l'inférence. C'est simplement l'utilisation
du modèle.

Pour être cohérent avec mon apprentissage, il me faut traiter l'information des images de la même manière que ce que
j'ai pu faire pour l'entrainement, ce bout de code ne va donc pas vous déboussoler : 

```python
# Load current image
image = cv2.imread(os.path.join(path, filename))

# Transform in gray 
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Separate the letters
for i in range(4):
    X.append(image[0:35, (i * 20):((i + 1) * 20)].flatten())

X = np.array(X, dtype=np.float) / 255.0 # Normalization
```

Il ne reste plus qu'à prédire les labels et les fusionner pour récupérer le captcha final

```python
prediction = model.predict(X)
letters = labels.inverse_transform(prediction)
captcha = "".join(letters)
```

Après avoir enregistré les résultats dans un CSV pour tous les captcha de niveau 1, on obtient plus de 50% de réussite
sur l'ensemble du jeu ! Les niveau 1 représente 62% de l'ensemble des données, la classification du niveau 1 marche 
drôlement bien :)

## Pourquoi ne pas utiliser Tesseract ?

Il faut noté qu'il est intéressant de tester les capacités du logiciel `Tesseract` sur ce challenge. Tesseract est un
logiciel de reconnaissance de caractère (OCR pour Optical Character Recognition) très performant qui est souvent 
utiliser pour numériser des documents scannés. Après installation on peut tester facilement en ligne de commande :

```bash
tesseract data/test/1.level1.png stdout
```

A noter, après une analyse du jeu d'entrainement, toutes les lettres ne sont pas présentes. Et il est possible de
configurer Tesseract pour ne reconnaitre que certaines lettres

```bash
tesseract data/test/1.level1.png stdout -c tessedit_char_whitelist=012345689ACDEFHKLMPQRSTUVXYZ
```

Pour réaliser l'ensemble du jeu de test, il est possible d'utiliser tesseract avec `pytesseract`


Le jeu reste en ligne pendant 1 semaine, l'occasion pour vous d'experimenter un peu plus et de monter dans le classement !
A bientôt pour un futur challenge ;)


