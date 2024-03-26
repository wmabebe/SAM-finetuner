#!/bin/bash

#export PATH=/share/apps/python/miniconda4.12/bin:/share/apps/gcc/9.1.0/bin:/people/scicons/deception/bin:/qfs/people/abeb563/.vscode-server/bin/b3e4e68a0bc097f0ae7907b217c1119af9e03435/bin/remote-cli:/people/scicons/deception/bin:/share/apps/python/miniconda4.12/condabin:/usr/lib64/qt-3.3/bin:/people/scicons/deception/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/opt/ganglia/bin:/opt/ganglia/sbin:/opt/pdsh/bin:/opt/rocks/bin:/opt/rocks/sbin:/people/abeb563/bin:/people/abeb563/bin:/people/abeb563/.local/bin
export PATH=$PATH:/people/abeb563/.local/bin

#module load python/miniconda4.12
module purge
module load python/miniconda4.12

source /share/apps/python/miniconda4.12/etc/profile.d/conda.sh

module load cuda/11.7
module load gcc/9.1.0


#python trainer.py

#python hugging.py

python3 finetune.py