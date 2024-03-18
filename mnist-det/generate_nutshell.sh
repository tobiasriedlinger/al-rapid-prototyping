python3 generate_dataset.py --split="val" --ds-size=500 --target-dir=/home/USR/active_learning_od/nutshell_mnist 
python3 generate_dataset.py --split="train" --ds-size=100000 --target-dir=/home/USR/active_learning_od/nutshell_mnist &
python3 generate_dataset.py --split="test" --ds-size=2000 --target-dir=/home/USR/active_learning_od/nutshell_mnist &