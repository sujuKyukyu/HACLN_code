# HACLN_code
This is the official code repository for the paper
“Enhancing Lung Cancer Prognostic Prediction via Hypergraph Aggregation and Contrastive Learning of Tumor Microenvironment” HACLN constructs a patient-centered hypergraph over multiplex immunofluorescence (MIF) images and applies a supervised contrastive loss across patient representations to improve prognostic accuracy.
## Requirements and Dependencies<br>
• Python 3.7 or higher<br>
• PyTorch (version 1.10.0) and torchvision (version 0.11.0) – see http://pytorch.org<br>
• scikit-learn (>=0.24), numpy, pandas, pyyaml, matplotlib, openpyxl<br>
Tip: If you later decide you want a one-step install, you can manually create a requirements.txt in the project root.<br> Then run:<br>
```pip install -r requirements.txt```
## Configuration
Edit ```config/config.yaml``` to set:<br>
• data_root – path to your feature folder<br>
• hypergraph.K_neigs – number of neighbors for KNN hyperedge construction<br>
• model.n_hid, training.lr, etc. – your model and training hyperparameters<br>
• output.result_root – directory for logs, checkpoints, and plots<br>

## Training
Run the main training and evaluation script with your config:
```python HACLN.py --config config/config.yaml```
This will:<br>
• Build the hypergraph from your features<br>
• Train the HACLN model <br>
• Validate each epoch, saving the best checkpoint to result/hgnn/ckpt<br>
• Print final metrics at completion<br>
## Acknowledgment
Appreciate the works and code repositories of those who came before:<br>
[Feng,Y., You, H., Zhang,Z., Ji, R., Gao,Y.:Hypergraph neural networks. In: Proceed-ings of the AAAI Conference on ArtifcialIntelligence, vol.33,pp.3558-3565(2019).](https://arxiv.org/abs/1809.09401)<br>
[G. Jaume et al., "Quantifying Explainers of Graph Neural Networks in Computational Pathology," 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Nashville, TN, USA, 2021, pp. 8102-8112.](https://ieeexplore.ieee.org/document/9577985)



