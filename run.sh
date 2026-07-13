#!/bin/bash -l
# L'argument '-l' est indispensable pour bénéficier des directives de votre .bashrc

# On peut éventuellement placer ici les commentaires SBATCH permettant de définir les paramètres par défaut de lancement :
#SBATCH --gres gpu:2
#SBATCH --mail-type FAIL,END
#SBATCH --time 3-00:00:00
# Si besoin, spécification de la version CUDA mise à disposition sur le cluster
# setcuda 11.7

# Exécution du script habituellement utilisé, on utilise la variable CUDA_VISIBLE_DEVICES qui contient la liste des GPU logiques actuellement réservés (toujours à partir de 0)

setcuda 12.3

conda activate myenv

python run_training.py -a "mftmodel"
