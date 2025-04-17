cd FlowMDM
python -m spacy download en_core_web_sm
pip install gdown
bash runners/prepare/download_smpl_files.sh
bash runners/prepare/download_glove.sh
bash runners/prepare/download_t2m_evaluators.sh
bash runners/prepare/download_pretrained_models.sh
cd ..
#- Download the processed version [here](https://drive.google.com/file/d/18a4eRh8mbIFb55FMHlnmI8B8tSTkbp4t/view?usp=share_link), and place it at `./dataset/babel`.
#- Download the following [here](https://drive.google.com/file/d/1PBlbxawaeFTxtKkKDsoJwQGuDTdp52DD/view?usp=sharing), and place it at `./dataset/babel`.
# reference: https://github.com/BarqueroGerman/FlowMDM/tree/main/runners