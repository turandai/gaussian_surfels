# High-quality Surface Reconstruction using Gaussian Surfels
<!-- [Pinxuan Dai](https://turandai.github.io/)\*, 
[Jiamin Xu](https://superxjm.github.io/)\*, 
Wenxiang Xie, 
[Xinguo Liu](http://www.cad.zju.edu.cn/home/xgliu),
[Huamin Wang](https://wanghmin.github.io/index.html),
[Weiwei Xu](http://www.cad.zju.edu.cn/home/weiweixu/index.htm)<sup>†</sup> -->

| [Project](https://turandai.github.io/projects/gaussian_surfels/) 
| [Paper](https://arxiv.org/pdf/2404.17774) 
| [arXiv](https://arxiv.org/abs/2404.17774) 
| [Data](https://huggingface.co/collections/turandai/gaussian-surfel-datasets-662cf9e2137b72821642add4) |<br>

The code builds upon the fantastic [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) code base and borrows the data preprocessing/loading part from [IDR](https://github.com/lioryariv/idr).


## Environment Setup
We did our experiments on Ubuntu 22.04.3, CUDA 11.8, and conda environment on Python 3.7.

Clone this repository:
```shell
git clone https://github.com/turandai/gaussian_surfels.git
cd gaussian_surfels
```

Create conda environment:
```shell
conda env create --file environment.yml
conda activate gaussian_surfels
```

If you need to recompile and reinstall the CUDA rasterizer:
```shell
cd submodules/diff-gaussian-rasterization
python setup.py install && pip install .
```

## Data Preparation
We test our method on subsets of on [DTU](https://roboimagedata.compute.dtu.dk/?page_id=36) and [BlendedMVS](https://github.com/YoYo000/BlendedMVS) datasets. 
We select 15 scenes from DTU and 18 scenes from BlendedMVS, then preprocess and normalize the data following [IDR](https://github.com/lioryariv/idr) data convention.
We also adopt [Omnidata](https://github.com/EPFL-VILAB/omnidata) to generate monocular normal prior.
You can download the data from [here](https://unimelbcloud-my.sharepoint.com/:f:/g/personal/xwwu1_student_unimelb_edu_au/EsdTFxnWonBMojuDmRorijkBCzvmAWA3eaRTn1q4M0axiQ?e=calgJN).


To test on your own unposed data, we recommend to use [COLMAP](https://github.com/colmap/colmap) for SfM initialization. To estimate monocular normal for your own data, please follow [Omnidata](https://github.com/EPFL-VILAB/omnidata) for additional environment setup. Download the pretrained model and run the normal estimation by:
```shell
cd submodules/omnidata
sh tools/download_surface_normal_models.sh
python estimate_normal.py --img_path path/to/your/image/directory
```

Note that precomputed normal of forementioned scenes from DTU and BlendedMVS are included in the downloaded dataset, so you don't have to run the normal estimation for them.


## Training
To train a scene:
```shell
python train.py -s path/to/your/data/directory
```
Trained model will be save in ```output/```.
To render images and reconstruct mesh from a trianed model:
```shell
python render.py -m path/to/your/trained/model --img --depth 10
```
We use screened Poisson surface reconstruction to extract mesh, at this line in ```render.py```:
```python
poisson_mesh(path, wpos, normal, color, poisson_depth, prune_thrsh)
```
In your output folder, ```xxx_plain.ply``` is the original mesh after Poisson reconstruction with the default depth of 10. For scenes in larger scales, you may use a higher depth level. For a smoother mesh, you may decrease the depth value.
We prune the Poisson mesh with a certain threshold to remove outlying faces and output ```xxx_pruned.ply```. This process sometimes may over-prune the mesh and results in holes. You may increase the "prune_thrsh" parameter accordingly.

## Evalutation
To evaluate the geometry accuracy on DTU, you have to download the [DTU](https://roboimagedata.compute.dtu.dk/?page_id=36) ground truth point cloud. 
For BlendedMVS evaluation, we fused, denoised and normalized the ground truth multi-view depth maps to a global point cloud as the ground truth geometry, which is included in our provided dataset for download. 
We follow previous work to use [this](https://github.com/jzhangbs/DTUeval-python) code to calculate the Chamfer distance between the ground truth point cloud and the reconstructed mesh.
```shell
# DTU evaluation:
python eval.py --dataset dtu --source_path path/to/your/data/directory --mesh_path path/to/your/mesh --dtu_gt_path path/to/DTU/MVSData --dtu_scanId ID
# BlendedMVS evaluation:
python eval.py --dataset bmvs --source_path path/to/your/data/directory --mesh_path path/to/your/mesh
```

<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">BibTeX</h2>
    <pre><code>@inproceedings{Dai2024GaussianSurfels,
  author = {Dai, Pinxuan and Xu, Jiamin and Xie, Wenxiang and Liu, Xinguo and Wang, Huamin and Xu, Weiwei},
  title = {High-quality Surface Reconstruction using Gaussian Surfels},
  publisher = {Association for Computing Machinery},
  booktitle = {ACM SIGGRAPH 2024 Conference Papers},
  year = {2024},
  articleno = {22},
  numpages = {11}
}</code></pre>
  </div>
</section>
