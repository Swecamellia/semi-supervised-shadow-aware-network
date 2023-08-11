# semi-supervised-shadow-aware-network
We propose a novel semi-supervised shadow aware network with boundary refinement (SABR-Net) to perform ultrasound images segmentation, which always have the challengs about the presence of shadow artifacts.
# Usage
It is highly recommanded to adopt Conda/MiniConda to manage the environment to avoid some compilation errors.
1. Clone the repository

 `git clone https://github.com/Swecamellia/semi-supervised-shadow-aware-network.git`

2. Install the dependencies
 * Python 3.9
 * Pytorch 1.11.0
 * Cuda 11.3
 * Other packages

`pip install -r requirements.txt`

# Datasets
 We evaluated the proposed semi-supervised US segmentation method using two publicly accessible US image datasets. The first dataset, CAMUS [1], is a multi-structure
cardiac dataset for US segmentation comprising clinical examinations of 500 patients. The second dataset, TN-SCUI [2], contains 3644 thyroid nodules of 3644 patients,
manually annotated by experienced radiologists. 

[1] S. Leclerc, E. Smistad, J. Pedrosa, A. Østvik, F. Cervenansky, F. Espinosa, T. Espeland, E. A. R. Berg, P.-M. Jodoin, T. Grenier, et al., “Deep learning for segmentation using an open large-scale dataset in 2d echocardiography,” IEEE Trans. Med. Imag., vol. 38, no. 9, pp. 2198–2210, 2019.

[2] H. Gireesha and S. Nanda, “Thyroid nodule segmentation and classification in ultrasound images,” Int. Journal of Engineering Research and Technology, 2014.

# Training 
  ### Steps 1: Pretrain
1. python **prepare_inception.py**:

   First prepare inception features for own dataset.
2. python **train_seg_pretrain.py**：
   
   Divide data of different proportions and mask the images, implement the inpainting and segmentation tasks of labeled and unlabeled data.

  ### Steps 2: Finetune

   python **train_seg_finetune.py**:

   To brige the gap between US images processed with shadow imitation operation and real US images.

  ### Steps 3: Optimzation & Inference
   python **test.py**

 # Inference
  Results Obtained on Multi Structure Cardiac and Thyroid Ultrasound Datasets
  
  
     


