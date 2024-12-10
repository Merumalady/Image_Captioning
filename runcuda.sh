#!/bin/bash
#SBATCH -n 4
#SBATCH -N 1
#SBATCH -D /export/fhome/vlia04/MyVirtualEnv
#SBATCH -t 4-00:05           # Tiempo máximo de ejecución en formato correcto
#SBATCH -p tfg               # Partición
#SBATCH --mem 12288          # Memoria en MB (12GB)
#SBATCH -o %x_%u_%j.out      # Archivo para la salida estándar
#SBATCH -e %x_%u_%j.err      # Archivo para los errores
#SBATCH --gres=gpu:1         # Solicitar 1 GPU correctamente

sleep 3

# Ejecución del script de Python
python /export/fhome/vlia04/MyVirtualEnv/Image_Captioning/seq2seqbasic.py