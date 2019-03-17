Plusieurs modèles ont été implémentés pour prédire la variable count : un gradient boosting et une régression RIDGE.

Les données étaient initialement nettoyées, j'ai constaté très peu de données manquantes, permettant ainsi une exploitation et une analyse plus simple de celles-ci. 

Dans le fichier "Exploratory_Data_Analysis" sont implémentées 3 classes : "Features_Plotting" qui permet de traiter et tracer les features selon si elles sont continues ou catégorielles, "Plotter" qui permet de tracer des graphes moustaches ainsi qu'une matrice de corrélation sous forme de heatmap, "DF_Modifier" qui permet de modifier le dataframe initial en ajoutant l'année, le mois, le jour et l'heure comme variables quantitatives ainsi que de réorganiser les colonnes pour améliorer les tracés qui suivent.

Le fichier "Modele" présente la classe "Model" permettant un traitement de données adéquat ainsi que l'implémentation des 2 modèles.

Le rapport final se présente sous la forme d'un Jupyter Notebook ("Probayes_Benjamin_SALEM.ipynb") avec une version de Python 3.7.2 (selon moi, jusqu'à 3.5 devrait fonctionner), toutes les cellules sont déjà run, vous pouvez tout relancer par vous-même, attention tout de même à l'entraînement avec recherche sur grille et cross-validation du gradient boosting qui prend entre 5 et 10 minutes pour tourner.

Il est possible, que ce soit dans le rapport ou dans le code, qu'il y ait un mix entre français et anglais, la plupart des termes techniques anglais étant utilisés ainsi en français.

Vous trouverez un fichier texte requirements.txt avec les versions des packages que j'ai utilisé.

# Setup Local
## create virtualenv
Link: <https://github.com/pyenv/pyenv>
    
	$ pyenv virtualenv 3.7.2 Probayes
	$ pyenv activate Probayes
## install libraries
    $ pip install --upgrade -r requirements.txt
## deactivate
    $ pyenv deactivate
