# -sb [Directory where the experiments are saved]
# -d  [Directory where the datasets are saved] 
# -e  [Experiment group]
# -c  [c=1: cuda, c=0: cpu] 

#exp=("convex_adaptive")
#sb="./results/quadra_convex"
#c=0
#d="./"

exp=('ijcnn_noniid_adaptive')
sb="./results/ijcnn"      
c=0
d="./datasets/libsvm/ijcnn"

#exp=('polyhedron')
#sb="./results/polyhedron"
#c=0
#d="./"


#exp=('cifar10_sdane')
#sb="./results/cifar10"   
#c=1   
#d="./datasets/cifar10"


python trainval.py \
-e=$exp \
-sb=$sb \
-d=$d \
-c=$c \