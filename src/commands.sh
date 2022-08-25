python train.py --max_iteration 6050 --data_dir /datasets/office31 --method DANN --source amazon --target webcam --out_dir dann_v1
python train.py --max_iteration 6050 --data_dir /datasets/office31 --method DANN --source amazon --target dslr --out_dir dann_v1
python train.py --max_iteration 6050 --data_dir /datasets/office31 --method DANN --source dslr --target webcam --out_dir dann_v1

python train.py --max_iteration 6050 --data_dir /datasets/office31 --method CDAN --source amazon --target webcam --out_dir cdan_v1
python train.py --max_iteration 6050 --data_dir /datasets/office31 --method CDAN --source amazon --target dslr --out_dir cdan_v1
python train.py --max_iteration 6050 --data_dir /datasets/office31 --method CDAN --source dslr --target webcam --out_dir cdan_v1

#### Evaluate
python eval.py --data_dir /datasets/office31 --target dslr --checkpoint ./snapshot/office31/dann_v1/amazon-dslr/best_model.pth --save ./snapshot/office31/dann_v1/amazon-dslr/eval.txt
python eval.py --data_dir /datasets/office31 --target webcam --checkpoint ./snapshot/office31/dann_v1/amazon-webcam/best_model.pth --save ./snapshot/office31/dann_v1/amazon-webcam/eval.txt
python eval.py --data_dir /datasets/office31 --target webcam --checkpoint ./snapshot/office31/dann_v1/dslr-webcam/best_model.pth --save ./snapshot/office31/dann_v1/dslr-webcam/eval.txt

python eval.py --data_dir /datasets/office31 --target dslr --checkpoint ./snapshot/office31/cdan_v1/amazon-dslr/best_model.pth --save ./snapshot/office31/cdan_v1/amazon-dslr/eval.txt
python eval.py --data_dir /datasets/office31 --target webcam --checkpoint ./snapshot/office31/cdan_v1/amazon-webcam/best_model.pth --save ./snapshot/office31/cdan_v1/amazon-webcam/eval.txt
python eval.py --data_dir /datasets/office31 --target webcam --checkpoint ./snapshot/office31/cdan_v1/dslr-webcam/best_model.pth --save ./snapshot/office31/cdan_v1/dslr-webcam/eval.txt