## Leaderboard on various pedestrian datasets
- If you find a new paper about pedestrian detection, please feel free to contact us for adding it.  
- If you find any error about performance, please feel free to contact us to fix it.  



## Table of Contents
1. [Caltech test set](#1)  
2. [Citypersons validation set](#2)  
3. [Citypersons test set](#3)  
4. [KITTI test set](#4)  
5. [KAIST test set](#5)

#### Caltech test set <a name="1"></a>
   
|    Method       | publication  | CNN |  **R**  | **HO** | **R+HO** | **A**| link |
| :--------      | :-----: | :-----: | :-------: | :-----: | :------: | :------: | :------: |
|  ACF            | PAMI2014   | no  |  44.2 |   90.2  |    54.6     |     79.6       | [Paper](https://vision.cornell.edu/se3/wp-content/uploads/2014/09/DollarPAMI14pyramids_0.pdf) |
|  SpatialPooling | ECCV2014   | no  |  29.2 |   84.1  |    41.7.6     |     74.0       | [Paper](https://arxiv.org/pdf/1409.5209.pdf)|
|  LDCF           | NIPS2014   | no  |  24.8 |   81.3  |   37.7     |     71.2       | [Paper](https://papers.nips.cc/paper/5419-local-decorrelation-for-improved-pedestrian-detection.pdf)|
|  Katamari       | ECCV2014   | no  |  22.5 |   84.4  |    36.2     |     71.3       | [Paper](https://arxiv.org/pdf/1411.4304.pdf)|
|  DeepCascade    | BMVC2015   | yes  |  31.1 |   81.7  |    42.4     |     74.1       |  [Paper](https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/43850.pdf) |
|  SCCPriors      | BMVC2015   | no  |  21.9 |   80.9  |    35.1     |     70.3       | [Paper](http://www.bmva.org/bmvc/2015/papers/paper176/paper176.pdf)  |
|  TA-CNN         | CVPR2015   | no  |  20.9 |   70.4  |   33.3    |     71.2       | [Paper](https://www.ee.cuhk.edu.hk/~xgwang/papers/tianLWTcvpr15.pdf)  |
|  CCF            | ICCV2015   | yes |  18.7 |   72.4  |    30.6     |     66.7       | [Paper](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Yang_Convolutional_Channel_Features_ICCV_2015_paper.pdf)  |
|  Checkerboards  | ICCV2015   | yes  | 18.5 |   77.5  |    31.8     |     68.7      |  [Paper](https://arxiv.org/abs/1501.05759.pdf) |
|  DeepParts      | ICCV2015   | yes  |  11.9|   60.4  |    22.8     |     64.8       | [Paper](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Tian_Deep_Learning_Strong_ICCV_2015_paper.pdf)   |
|  CompACT-Deep   | BMVC2015   | yes  |  11.7 |   65.8  |    24.6     |     64.4       | [Paper](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Cai_Learning_Complexity-Aware_Cascades_ICCV_2015_paper.pdf)  |
|  NNNF           | CVPR2016   | no  |  16.2 |  74.9  |    -     |     -       | [Paper](https://arxiv.org/pdf/1511.08058.pdf)  |
|  MS-CNN         | ECCV2016   | yes  |  10.0 |  59.9  |    21.5     |     60.9       | [Paper](https://arxiv.org/pdf/1607.07155.pdf)  |
|  RPN+BF         | ECCV2016   | yes  |  9.6 |   74.3  |    24.0     |    64.7       | [Paper](https://arxiv.org/pdf/1607.07032.pdf)  |
|  MCF            | TIP2017    | yes  |  10.4 |  66.7  |    -     |     -       | [Paper](https://arxiv.org/abs/1603.00124.pdf)  |
|  F-DNN          | WACV2017   | yes  |  8.6 |  55.1  |    19.3     |     50.6       | [Paper](https://arxiv.org/pdf/1610.03466.pdf)  |
|  PCN            | BMVC2017   | yes  |  8.4 |   55.8  |    19.2    |     61.9     | [Paper](https://arxiv.org/pdf/1804.04483.pdf)  |
|  PDOE           | ECCV2018   | yes  |  7.6|   44.4  |    -     |     -       | [Paper](https://openaccess.thecvf.com/content_ECCV_2018/papers/CHUNLUAN_ZHOU_Bi-box_Regression_for_ECCV_2018_paper.pdf)  |
|  UDN+           | PAMI2018   | yes  |  11.5 |   70.3  |    24.7     |     64.8       |  [Paper](https://wlouyang.github.io/Papers/Ouyang2017JoingCNNPed.pdf) |
|  FRCNN+ATT      | CVPR2018   | yes  |  10.3 |   45.2  |    18.2     |     54.5       | [Paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Occluded_Pedestrian_Detection_CVPR_2018_paper.pdf)  |
|  SA-FRCNN       | TMM2018   |  yes |  9.7 |   64.4  |    21.9     |     62.6       | [Paper](https://arxiv.org/pdf/1510.08160.pdf)  |
|  ADM            | TIP2018   |  yes |  8.6 |   30.4  |   13.7     |     42.3       | [Paper](https://arxiv.org/pdf/1602.01237.pdf)  |
|  GDFL           | ECCV2018   | yes  |  7.8 |   43.2 |    15.6     |     48.1       |  [Paper](https://openaccess.thecvf.com/content_ECCV_2018/papers/Chunze_Lin_Graininess-Aware_Deep_Feature_ECCV_2018_paper.pdf) |
|  TLL-TFA        | ECCV2018   | yes  |  7.4 |   28.7  |    12.3     |     38.2       | [Paper](https://openaccess.thecvf.com/content_ECCV_2018/papers/Tao_Song_Small-scale_Pedestrian_Detection_ECCV_2018_paper.pdf)  |
|  AR-Ped         | CVPR2019   | yes  |  6.5 |   48.8  |    16.1    |     58.9       |  [Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Brazil_Pedestrian_Detection_With_Autoregressive_Network_Phases_CVPR_2019_paper.pdf) |
|  FRCNN+A+DT     | ICCV2019   | yes  |  8.0 |   37.9  |   -    |     -      | [Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Zhou_Discriminative_Feature_Transformation_for_Occluded_Pedestrian_Detection_ICCV_2019_paper.pdf)  |
|  MGAN           | ICCV2019   | yes  |  6.8 |   38.1  |    13.8     |     -      |  [Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Pang_Mask-Guided_Attention_Network_for_Occluded_Pedestrian_Detection_ICCV_2019_paper.pdf) |
|  TFAN           | CVPR2020   | yes  |  6.7 |   30.9  |    12.4     |     -       | [Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wu_Temporal-Context_Enhanced_Detection_of_Heavily_Occluded_Pedestrians_CVPR_2020_paper.pdf)  |


|    Method       | publication  | CNN |  **R**  | **HO** | **R+HO** | **A**| link |
| :--------      | :-----: | :-----: | :-------: | :-----: | :------: | :------: | :------: |
|  HyperLearner   | CVPR2017   | yes  |  5.5 |   - |    -     |     -       | [Paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Mao_What_Can_Help_CVPR_2017_paper.pdf)  |
|  RepLoss        | CVPR2018   | yes  |  4.0 |   - |    -    |    -       | [Paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Repulsion_Loss_Detecting_CVPR_2018_paper.pdf)  |
|  ALFNet         | ECCV2018   | yes  |  4.5 |  - |    -    |     -       | [Paper](https://openaccess.thecvf.com/content_ECCV_2018/papers/Wei_Liu_Learning_Efficient_Single-stage_ECCV_2018_paper.pdf)  |
|  BGRNet   | ACM-MM2020   | yes  |  4.5 |   - |    -    |     -      |  [Paper](https://dl.acm.org/doi/pdf/10.1145/3394171.3413989)  | 
|  OR-CNN         | ECCV2018   | yes  |  4.1 |   -  |    -     |     -       | [Paper](https://openaccess.thecvf.com/content_ECCV_2018/papers/Shifeng_Zhang_Occlusion-aware_R-CNN_Detecting_ECCV_2018_paper.pdf)  |
|  HGPD  |  ACM-MM2020    |  yes |   3.78  |     -    |      -     |     -      |  [Paper](https://dl.acm.org/doi/pdf/10.1145/3394171.3413983) |
|  SML        | ACMMM2020   | yes  | 3.7 |   -  |    -     |     -       | [Paper](https://cse.buffalo.edu/~jsyuan/papers/2020/SML.pdf)  |
|  JointDet       | AAAI2020   | yes  |  3.0 |  -  |    -     |     -       | [Paper](https://arxiv.org/pdf/1909.10674.pdf)  |
|  PedHutter      | AAAI2020   | yes  |  2.3 |   - |    -     |     -       | [Paper](https://www.aaai.org/Papers/AAAI/2020GB/AAAI-ChiC.961.pdf)  |

- The top table uses the original annotations, while the bottom table uses the new annotations.
- CNN indicates whether or not deep features are used.


#### Citypersons validation set <a name="2"></a>

|    Method       | publication  | scale |  **R**  | **HO** | link |
| :--------      | :-----: | :-----: | :-------: | :-----: | :-----: |
|  Adapted FRCNN  | CVPR2017   | 1.0x  | 15.4|   -  |  [Paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Zhang_CityPersons_A_Diverse_CVPR_2017_paper.pdf)   |  
|  RepLoss       | CVPR2018   | 1.0x  |  13.7 |   56.9†  | [Paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Repulsion_Loss_Detecting_CVPR_2018_paper.pdf)  |  
|  FRCNN+ATT           | CVPR2018   |  1.0x | 16.0 |   56.7  | [Paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Occluded_Pedestrian_Detection_CVPR_2018_paper.pdf)   |  
|  TLL+MRF       | ECCV2018   | 1.0x  |  14.4 |   52.0†  |  [Paper](https://openaccess.thecvf.com/content_ECCV_2018/papers/Tao_Song_Small-scale_Pedestrian_Detection_ECCV_2018_paper.pdf)   |  
|  OR-CNN    | ECCV2018   | 1.0x  |  12.8 |   55.7†  |  [Paper](https://openaccess.thecvf.com/content_ECCV_2018/papers/Shifeng_Zhang_Occlusion-aware_R-CNN_Detecting_ECCV_2018_paper.pdf)   |  
|  ALFNet      | ECCV2018   | 1.0x  |  12.0 |  51.9†  | [Paper](https://openaccess.thecvf.com/content_ECCV_2018/papers/Wei_Liu_Learning_Efficient_Single-stage_ECCV_2018_paper.pdf)  |  
|  Cascade RCNN        | CVPR2018   | 1.0x  |  12.0 |   49.4  |  [Paper](https://arxiv.org/abs/1712.00726.pdf)  |  
|  LBST      | TIP2019   | 1.0x  |  12.6 |  48.7  | [Paper](https://ieeexplore.ieee.org/abstract/document/8931263/)  |  
|  CSP            | CVPR2019   | 1.0x |  11.0 |   49.3†  |  [Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_High-Level_Semantic_Feature_Detection_A_New_Perspective_for_Pedestrian_Detection_CVPR_2019_paper.pdf)   |  
|  Adaptive-NMS  | CVPR2019   | 1.0x  | 11.9 |   55.2 † | [Paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_Adaptive_NMS_Refining_Pedestrian_Detection_in_a_Crowd_CVPR_2019_paper.pdf)  |  
|  MGAN      | ICCV2019   | 1.0x  |  11.3|   42.0 |  [Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Pang_Mask-Guided_Attention_Network_for_Occluded_Pedestrian_Detection_ICCV_2019_paper.pdf)   |  
|  R2NMS   | CVPR2020   | 1.0x  |  11.1 |   53.3†  |  [Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Huang_NMS_by_Representative_Region_Towards_Crowded_Pedestrian_Detection_by_Proposal_CVPR_2020_paper.pdf)  | 
|  PRNet   | ECCV2020   | 1.0x  |  10.8 |   42.0  |  [Paper](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123680035.pdf)  | 
|  CaSe   | ECCV2020   | 1.0x  |  10.5 |   40.5  |  [Paper](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123620086.pdf)  | 
|  BGRNet   | ACM-MM2020   | 1.0x  |  9.4 |   45.9†  |  [Paper](https://dl.acm.org/doi/pdf/10.1145/3394171.3413989)  | 
|  Adapted FRCNN  | CVPR2017   |  1.3x |  12.8 |   - |  [Paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Zhang_CityPersons_A_Diverse_CVPR_2017_paper.pdf)  |  
|  RepLoss       | CVPR2018   |  1.3x |  11.6 |   55.3†  |  [Paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Repulsion_Loss_Detecting_CVPR_2018_paper.pdf) |  
|  OR-CNN    | ECCV2018   | 1.3x  |  11.0 |   51.3†  |  [Paper](https://openaccess.thecvf.com/content_ECCV_2018/papers/Shifeng_Zhang_Occlusion-aware_R-CNN_Detecting_ECCV_2018_paper.pdf)   |  
|  PDOE      | ECCV2018   | 1.3x  |  11.2 |   44.2  | [Paper](https://openaccess.thecvf.com/content_ECCV_2018/papers/CHUNLUAN_ZHOU_Bi-box_Regression_for_ECCV_2018_paper.pdf)   | 
|  LBST      | TIP2019   | 1.3x  |  11.4 |  45.2  | [Paper](https://ieeexplore.ieee.org/abstract/document/8931263/)  |  
|  Adaptive-NMS  | CVPR2019   | 1.3x  | 10.8 |   54.2 † | [Paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_Adaptive_NMS_Refining_Pedestrian_Detection_in_a_Crowd_CVPR_2019_paper.pdf)  | 
|  HGPD  | ACM-MM2020   | 1.3x  | 10.9 |   40.9  | [Paper](https://dl.acm.org/doi/pdf/10.1145/3394171.3413983)  |  
|  IoU+Sign  | ICIP2019   | 1.3x  | 10.8 |   54.3†  | [Paper](https://arxiv.org/abs/1911.11449.pdf)  |  
|  NOH-NMS  | ACM-MM2020   | 1.3x  | 10.8 |   53.0  | [Paper](https://arxiv.org/pdf/2007.13376.pdf)  |  
|  FRCNN+A+DT  | CVPR2019   | 1.3x  | 11.1 |   44.3  | [Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Zhou_Discriminative_Feature_Transformation_for_Occluded_Pedestrian_Detection_ICCV_2019_paper.pdf)  |  
|  SML      | ACMMM2020   | 1.3x  |  10.6|  -  |  [Paper](https://cse.buffalo.edu/~jsyuan/papers/2020/SML.pdf)   |  
|  MGAN      | ICCV2019   | 1.3x  |  10.5|  39.4  |  [Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Pang_Mask-Guided_Attention_Network_for_Occluded_Pedestrian_Detection_ICCV_2019_paper.pdf)   |  
|  CaSe   | ECCV2020   | 1.3x  |  9.8 |   37.4  |  [Paper](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123620086.pdf)  | 
|  JointDet  | CVPR2019   | 1.3x  | 10.2 |   -  |  [Paper](https://arxiv.org/pdf/1909.10674.pdf) |  
|  0.5-stage  | WACV2020   | 1.3x  | 8.1 |   -  | [Paper](https://openaccess.thecvf.com/content_WACV_2020/papers/Ujjwal_A_one-and-half_stage_pedestrian_detector_WACV_2020_paper.pdf)  |  
|  PedHutter  | AAAI2020   | 1.3x  | 8.3 |   43.5†  | [Paper](https://www.aaai.org/Papers/AAAI/2020GB/AAAI-ChiC.961.pdf)  |  

- Usually, **HO** represents pedestrians over 50 pixels in height with 35-80% occlusion. 
- † indicates the pedestrians over 50 pixels in height with more than 35% occlusion. Thus, † suggest higher difficulty.

#### Citypersons test set <a name="3"></a>
   
|    Method       | publication  |  **R**  | **RS** | **HO** | **A**| link |
| :--------      | :-----:  | :-------: | :-----: | :------: | :------: | :------: |
|  MS-CNN         | ECCV2016    |  13.32 |   15.86  |    51.88     |     39.94   |  [Paper](https://arxiv.org/pdf/1607.07155.pdf)   |
|  Adapted FRCNN  | CVPR2017    |  12.97 |   37.24  |    50.47     |     43.86   |  [Paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Zhang_CityPersons_A_Diverse_CVPR_2017_paper.pdf)   |
|  Cascade MS-CNN  | CVPR2018    |  11.62 |   13.64  |    47.14     |     37.63   |  [Paper](https://arxiv.org/abs/1712.00726.pdf)    |
|  RepLoss  | CVPR2018    |  11.48 |   15.67  |    52.59     |     39.17   |  [Paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Repulsion_Loss_Detecting_CVPR_2018_paper.pdf)   |
|  Adaptive-NMS  | CVPR2019    |  11.40 |   13.64  |    46.99     |     38.89   |  [Paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_Adaptive_NMS_Refining_Pedestrian_Detection_in_a_Crowd_CVPR_2019_paper.pdf)   |
|  OR-CNN  | ECCV2018    |  11.32 |   14.19  |    51.43     |     40.19   |  [Paper](https://openaccess.thecvf.com/content_ECCV_2018/papers/Shifeng_Zhang_Occlusion-aware_R-CNN_Detecting_ECCV_2018_paper.pdf)  |
|  MHN  | TMM2019    |  12.92 |   17.24  |    46.72     |     39.16   |  [Paper](https://ieeexplore.ieee.org/abstract/document/8887288/)  |
|  HBA-RCNN | -    |  11.26 |   15.68  |    39.54     |     38.77   |  - |
|  DVRNet  |  -    |  10.99 |   15.68  |    43.77     |     41.48   |  - |
|  HGPD  |  ACM-MM2020    |  10.17 |   -  |     38.65     |      38.24   |  [Paper](https://dl.acm.org/doi/pdf/10.1145/3394171.3413983) |
|  MGAN  | ICCV2019    |  9.29 |   11.38  |    40.97     |     38.86   | [Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Pang_Mask-Guided_Attention_Network_for_Occluded_Pedestrian_Detection_ICCV_2019_paper.pdf)
|  STNet  | -    |  8.92 |   11.13  |    34.31     |     29.54  | -  |
|  YT-PedDet | -    |  8.41 |   10.60 |    37.88     |     37.22   |  - |
|  APD | arXiv2019    |  8.27 |   11.03  |    35.45     |     35.65   | [Paper](https://arxiv.org/pdf/1910.09188.pdf)  |
|  Pedestron  | arXiv2020    |  7.69 |   9.16  |    27.08     |     28.33   |  [Paper](https://arxiv.org/pdf/2003.08799.pdf)  |

- **RS** represents the pedestrians over 50 pixels and under 75 pixels with less than 0.35 occlusion, while **A** the pedestrians over 20 pixels with
less than 0.8 occlusion.

#### KITTI test set <a name="4"></a>

   
|    Method       | publication  | Medium |  Easy  | Hard | link |
| :--------      | :-----: | :-----: | :-------: | :-----: | :-----: |
|  ACF              | PAMI2014   | 39.81  | 44.49|   37.21  |  [Paper](https://vision.cornell.edu/se3/wp-content/uploads/2014/09/DollarPAMI14pyramids_0.pdf) |
|  Checkerboards       | CVPR2015   | 56.75  |  67.65|   51.12  |  [Paper](https://arxiv.org/abs/1501.05759.pdf) |
|  DeepParts           | ICCV2015   |  58.67 | 70.49 |   52.78 |   [Paper](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Tian_Deep_Learning_Strong_ICCV_2015_paper.pdf) |
|  CompACT-Deep       | ICCV2015   | 58.74  |  70.69 |   52.71  | [Paper](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Cai_Learning_Complexity-Aware_Cascades_ICCV_2015_paper.pdf) |  
|  Regionlets    | PAMI2015   | 60.83  | 73.79 |   54.72  |  [Paper](http://users.eecs.northwestern.edu/~mya671/mypapers/ICCV13_Wang_Yang_Zhu_Lin.pdf) | 
|  NNNF      | CVPR2016   | 58.01  |  69.16 |  52.77 | [Paper](https://openaccess.thecvf.com/content_cvpr_2016/papers/Cao_Pedestrian_Detection_Inspired_CVPR_2016_paper.pdf) |
|  MCF        | TIP2016   | 59.45  |  70.87 |  54.28  |   [Paper](https://arxiv.org/pdf/1603.00124.pdf) |
|  RPN+BF            | ECCV2016   | 61.29 |  75.45 |   56.08  |   [Paper](https://arxiv.org/pdf/1607.07032.pdf)  |
|  SDP+RPN  | CVPR2016   | 70.42 | 82.07 |   65.09 | [Paper](https://openaccess.thecvf.com/content_cvpr_2016/papers/Yang_Exploit_All_the_CVPR_2016_paper.pdf) |
|  IVA      | ACCV2016   | 71.37  |  84.61|   64.90|   [Paper](https://link.springer.com/chapter/10.1007/978-3-319-54184-6_26) |
|  MS-CNN   | ECCV2016   | 74.89  | 85.71 |   68.99 |  [Paper](https://arxiv.org/pdf/1607.07155.pdf) |
|  SubCNN  | WACV2017   |  72.77 |  84.88|   66.82 |  [Paper](https://arxiv.org/pdf/1604.04693.pdf) |
|  PCN       | BMVC2017   |  63.41 |  80.08 |   58.55 |  [Paper](https://arxiv.org/pdf/1804.04483.pdf) |
|  GN    | PRL2017   | 72.29  | 82.93 |   65.56  |   [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0167865517300545) |
|  RRC      | CVPR2017   | 76.61 |  85.98 |   71.47  |  [Paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Ren_Accurate_Single_Stage_CVPR_2017_paper.pdf) |
|  CFM  | TCSVT2018   | 62.84  | 74.76 |   56.06  | [Paper](https://arxiv.org/pdf/1603.04525.pdf) |
|  SAF R-CNN  | TMM2018   | 65.01  | 77.93 |   60.42  | [Paper](https://arxiv.org/pdf/1510.08160.pdf) |
|  SJTU-HW  | ICIP2018   | 75.81 | 87.17 |   69.86  | [Paper](http://resources.dbgns.com/study/ObjectDetection/NMS-LED.pdf) |
|  GDFL      | ECCV2018   | 68.62  |  84.61|  66.86  |   [Paper](https://openaccess.thecvf.com/content_ECCV_2018/papers/Chunze_Lin_Graininess-Aware_Deep_Feature_ECCV_2018_paper.pdf) |
|  MonoPSR  | CVPR2019   | 68.56  |  85.60 |   63.34 | [Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Bertoni_MonoLoco_Monocular_3D_Pedestrian_Localization_and_Uncertainty_Estimation_ICCV_2019_paper.pdf) |
|  FFNet  | PR2019   | 75.99  | 87.21 |   69.86  |  [Paper](https://arxiv.org/abs/1909.10970.pdf)
|  MHN  | TCSVT2019   | 75.99  | 87.21 |   69.50  | [Paper](https://ieeexplore.ieee.org/abstract/document/8887288/)
|  Aston-EAS  | TITS2019   | 76.07  | 86.71 |   70.02  |   [Paper](https://ieeexplore.ieee.org/document/8694965) 
|  AR-Ped  | CVPR2019   | 73.44  | 83.66 |   68.12  |   [Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Brazil_Pedestrian_Detection_With_Autoregressive_Network_Phases_CVPR_2019_paper.pdf)  | 

#### KAIST test set <a name="5"></a>

   
|    Method       | publication  | MR(All)|MR(Day)| MR(Nighy)| link |
| :--------      | :-----: | :-----: | :-------: | :-----: | :-----: |
|ACF | CVPR2015 |47.32 |42.57|56.17|[Paper](https://www.cvfoundation.org/openaccess/content_cvpr_2015/papers/Hwang_Multispectral_Pedestrian_Detection_2015_CVPR_paper.pdf)|
|Halfway Fusion | BMVC2016 |25.75 |24.88|26.59|[Paper](https://webpages.uncc.edu/~szhang16/paper/BMVC16_multispectral.pdf)|
|IAF-RCNN|PR2019|15.73|14.55|18.26|[Paper](https://www.sciencedirect.com/science/article/pii/S0031320318303030)|
|IATDNN+IAMSS|IF2019|14.95|14.67|15.72|[Paper](https://www.sciencedirect.com/science/article/pii/S1566253517308138)|
|CIAN|IF2019|14.12|14.77|11.13|[Paper](https://www.sciencedirect.com/science/article/pii/S1566253518304111)|
|MSDS-RCNN|BMVC2018|11.34|10.53|12.94|[Paper](https://arxiv.org/abs/1808.04818)|
|AR-CNN|ICCV2019|9.34|9.94|8.38|[Paper](https://arxiv.org/abs/1901.02645)|
|MBNet|ECCV2020|8.13|8.28|7.86|[Paper](https://arxiv.org/abs/2008.03043)|

