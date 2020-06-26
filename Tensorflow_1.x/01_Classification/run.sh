python3 ./dataset/generate_json.py 
python3 ./dataset/generate_dataset.py --dataset_name flower_photos

python3 train.py --dataset_name flower_photos --classes 5 --augmentation default
python3 create_pb_files.py --dataset_name flower_photos --classes 5 

