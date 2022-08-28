---
title: paper_UKD_2022
mathjax: true
date: 2022-06-18 17:13:52
tags:
  - ML
  - SSB
  - paper
categories:
  - MachineLearning
  - 搜广推
---

[paper](https://arxiv.org/pdf/2201.08024.pdf)
# 1 Background
- Sample Selection Bias
{% asset_img SSB.png SSB %}

<!-- more -->

# 2 The representative methods
## 2.0 Base CVR Models Trained in Click Space
> joint training, 最简单的做法
- ctr/cvr共享底层，利用ctr任务来1）缓解cvr数据稀疏；2）间接缓解SSB问题
$$
\min _{F_{v}, F_{c}} \frac{1}{\left|\mathcal{D}_{\text {click }}\right|} \sum_{\mathcal{D}_{\text {click }}} \ell\left(y_{\text {conv }}, F_{v}(x)\right)+\gamma \frac{1}{|\mathcal{D}|} \sum_{\mathcal{D}} \ell\left(y_{c l i c k}, F_{c}(x)\right)
$$
- 数据分布不同，学到的embedding不准，进而影响cvr分数 **Thus their predicted CVR scores in unclicked space may have a non-negligible deviation because there is a discrepancy between the data distributions of click and unclick ads.**

## 2.1 auxiliary task learing
- ESMM: pv-conv: post view conv
$$
\min _{F_{v}, F_{c}} \frac{1}{|\mathcal{D}|} \sum_{\mathcal{D}}\left(\ell\left(y_{p v \text {-conv }}, F_{c}(x) \cdot F_{v}(x)\right)+\gamma \ell\left(y_{c l i c k}, F_{c}(x)\right)\right)
$$
- Limitations
  - 1）对于点击样本，如果其y pv-conv=0: 对CTCVR和CVR任务来说，分别是其负样本和正样本，两个学习任务会产生gradient conflict
  - 2）对于未点击样本：其y conv是unknown, 而模型会倾向于将其pcvr优化为0
{% asset_img esmm_limitation.png esmm_limitation %}

## 2.2 counterfactual learning
- 两类：inverse propensity score (IPS) and doubly robust (DR) estimators


# 3 Methods
> 两部分：1）用于学习click-adaptive的teacher model, 作用是为unclicked ads提供pseudo-conversion labels; 2) an uncertainty-regularized student model that is trained on entire impression space.

{% asset_img model.png model %}

## 3.1 click-adaptive teacher model
- 说明：
{% asset_img teacher.png teacher %}
- forward：impression ads
$$
\begin{aligned}
\boldsymbol{h}^{(T)} &=T_{f}(x) \\
\boldsymbol{p}_{\text {conv }}^{(T)} &=\operatorname{softmax}\left(T_{p}\left(\boldsymbol{h}^{(T)}\right)\right)=\left(\hat{p}_{C V R}^{(T)}, 1-\hat{p}_{C V R}^{(T)}\right) \\
\boldsymbol{p}_{d} &=\operatorname{softmax}\left(T_{d}\left(\boldsymbol{h}^{(T)}\right)\right)
\end{aligned}
$$
- backward：
  - 1）对于click样本，优化cvr predictor和representation learner提高其cvr predict效果 --> min cvr loss
  $$
  \min _{T_{f}, T_{p}} \mathcal{L}_{C V R}^{(T)}=\frac{1}{\left|\mathcal{D}_{\text {click }}\right|} \sum_{\mathcal{D}_{\text {click }}} \ell\left(y_{\text {conv }}, \boldsymbol{p}_{\text {conv }}^{(T)}\right)
  $$
  - 2）对于impression样本
    - a. click discriminator提高其ctr predict效果 --> min ctr loss
    - b. click-adaptive: 无法区分某个样本的representation是from either click domain or unclick domain --> max ctr loss --> gradient reversal。实验结果是ctr判别器的auc为0.5
  $$
  \max _{T_{f}} \min _{T_{d}} \mathcal{L}_{d}^{(T)}=\frac{1}{|\mathcal{D}|} \sum_{\mathcal{D}} \ell\left(y_{c l i c k}, \boldsymbol{p}_{d}\right)
  $$
- 作用：click discriminator不能区分来自click domain or unclick domain的representation --> cvr模型在unclick样本上可以获得同click样本一样可靠的效果 --> cvr模型为unclick样本打上pseudo conversion label

## 3.2 Uncertainty-Regularized Student Model
### 3.2.1 Base Student Model: Label Distillation
- 直接利用teacher打的pseudo conversion label做ctr和cvr的联合学习。
- Distilling Knowledge from Unclicked Ads，利用pseudo conversion labels可以解决SSB问题

$$
\mathcal{L}_{C V R}=\underbrace{\sum_{\mathcal{D}_{\text {click }}} \ell\left(y_{\text {conv }}, \boldsymbol{p}_{\text {conv }}\right)}_{\mathcal{L}_{C V R_{-} \text {click }}}+\alpha \underbrace{\sum_{\mathcal{D}_{\text {unclick }}} \ell\left(\boldsymbol{p}_{\text {conv }}^{(T)}, \boldsymbol{p}_{\text {conv }}\right)}_{\mathcal{L}_{C V R} \text { unclick }}
$$
- 整体loss为cvr与ctr loss的加权，ctr任务的作用在于通过share embedding的方式为cvr预估引入更多信息

$$
\mathcal{L}_{s t u d e n t}=\mathcal{L}_{C V R}+\gamma \mathcal{L}_{C T R}
$$
- 限制：效果依赖于teacher model的精度 --》 pseudo conversion labels不可能有ground truth的confidence --》设法减少pseudo conversion labels带来的噪声

### 3.2.2 Uncertainty-regularized Student: Alleviate Noise
- 1) 识别噪声，找到unreliable unclicked samples --》unreliable的样本其uncertainty越大
- 2) 降低1）找到的样本影响 --》loss加权

#### 3.2.2.1 Noise Samples Identification
- 一个简单的实验：Dclick数据集，随机选择k%比例的正样本，将其label置换为负样本（为保证正负样本比例，对负样本做同样操作），利用一个具有两个head的cvr模型预测两个cvr值，求两个cvr的kl散度
  - 实验结果表明，noise sample的两次cvr预测值kl散度较大
{% asset_img student_kl.png kl %}

- 因此有两个cvr predictor，求其获得的representation的kl散度，这里对两个predictor做了dropout来increase the discrepancy of two predictors
$$
\begin{array}{l}
\boldsymbol{p}_{\text {conv }}=S_{p}^{v}\left(\text { dropout }\left(\boldsymbol{h}_{\text {conv }}\right)\right), \boldsymbol{p}_{\text {conv }}^{\prime}=S_{p^{\prime}}^{v}\left(\text { dropout }^{\prime}\left(\boldsymbol{h}_{\text {conv }}\right)\right) \\
\mathrm{KL}\left(\boldsymbol{p}_{\text {conv }} \| \boldsymbol{p}_{\text {conv }}^{\prime}\right)=\boldsymbol{p}_{\text {conv }} \log \frac{\boldsymbol{p}_{\text {conv }}}{\boldsymbol{p}_{\text {conv }}^{\prime}}
\end{array}
$$
- 问题
  - noise很大的时候clean sample的kl散度也很大 --》KL散度loss以及加权需要调整？
  - 其他的noise识别方法

#### 3.2.2.2 Distillation with Uncertainty Regularization
- 1）对unclicked sample对其cvr loss用KL散度加权
$$
\sum_{\widetilde{D}_{\text {unclick }}} \exp \left(-\lambda \cdot \mathrm{KL}\left(\boldsymbol{p}_{\text {conv }} \| \boldsymbol{p}_{\text {conv }}^{\prime}\right)\right) \cdot \ell\left(\boldsymbol{p}_{\text {conv }}^{(T)}, \boldsymbol{p}_{c o n v}\right)
$$
- 2）添加uncertainty正则项
$$
\sum_{\widetilde{\mathcal{D}}_{\text {unclick }}} \mathrm{KL}\left(\boldsymbol{p}_{\text {conv }} \| \boldsymbol{p}_{\text {conv }}^{\prime}\right)
$$