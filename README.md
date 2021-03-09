# VectorAscent: Generate vector graphics from a textual description

## Example
"a painting of an evergreen tree"
```
python text_to_painting.py --prompt "a painting of an evergreen tree" --num_iter 2500 --use_blob --subdir vit_rn50_useblob
```
![a painting of an evergreen tree](iter_1960.svg)

We rely on [CLIP](https://arxiv.org/abs/2103.00020) for its aligned text and image encoders, and [diffvg](https://people.csail.mit.edu/tzumao/diffvg/), a differentiable vector graphics rasterizer. Differentiable rendering allows us to generate raster images from vector paths, but isn't provided textual descriptions. We use CLIP to score the similarity between *raster* graphics and textual captions. Using gradient ascent, we can then optimize for a vector graphic whose rasterization has high similarity with a user-provided caption, backpropagating through CLIP and diffvg to the vector graphics parameters. This project is partially inspired by [Deep Daze](https://twitter.com/advadnoun/status/1348375026697834496), a caption-guided raster graphics generator.

## Quick start
Requirements:
 - torch
 - torchvision
 - matplotlib
 - numpy
 - scikit-image
 - clip
 - diffvg


Install our dependencies and CLIP.
```
conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
pip install ftfy regex tqdm numpy matplotlib scikit-image
pip install git+https://github.com/openai/CLIP.git
```

Then follow these [instructions to install diffvg](https://github.com/BachiLi/diffvg).
