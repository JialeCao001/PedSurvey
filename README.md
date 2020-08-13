## From Handcrafted to Deep Features for Pedestrian Detection: A Survey

- This project provides a paper list about pedestrian detection following the taxonomy in our survey paper. 
- If you find a new paper about pedestrian detection, please feel free to contact us.  
- If this project help your research, please consider to cite our paper:
```
@ARTICLE{Cao_PDR_arXiv_2020,
         author = {Kemal Oksuz and Baris Can Cam and Sinan Kalkan and Emre Akbas},
         title = "{From Handcrafted to Deep Features for Pedestrian Detection: A Survey}",
         journal = {arXiv},
         year = "2020"
        }
```

## Table of Contents
1. [Detection pipeline](#1)  
    1.1 [Proposal generation](#1.1)  
    1.2 [Feature extraction](#1.2)   
    1.3 [Proposal classification](#1.3)  
    1.4 [Post processing](#1.4)  
2. [Single-spectral pedestrian detection](#2)  
    2.1 [Handcrafted features based method](#2.1)  
    2.2 [Deep features based method](#2.2)    
3. [Multispectral pedestrian detection](#3)  
    3.1 [Deep feature fusion](#3.1)  
    3.2 [Data processing](#3.2)  
    3.3 [Domain adaptation](#3.3) 
4. [Datasets](#4)  
    4.1 [Earlier pedestrian datasets](#4.1)  
    4.2 [Modern pedestrian datasets](#4.2)  
    4.3 [Multispectral pedestrian datasets](#4.3)  
5. [Challenges](#5)  
    5.1 [Scale variance](#5.1)  
    5.2 [Occlusion](#5.2)  
    5.3 [Domain adaptation](#5.3)  
        
## 1. Detection pipeline <a name="1"></a>

- Proposal generation <a name="1.1"></a>
  - Sliding windows
  - Objectness methods
    - Selective search for object recognition, IJCV 2016.
    - What makes for effective detection proposals, PAMI 2016. 
    - Bing: Binarized normed gradients for objectness estimation at 300fps, CVPR 2014. 
    - Edge boxes: Locating object proposals from edges, ECCV 2014.
  - Region proposal networks  
    - Faster rcnn: Towards real-time object detection with region proposal networks, NIPS 2015.
    - Region proposal by guided anchoring, CVPR 2019.
    - A unified multi-scale deep convolutional neural network for fast object detection, ECCV 2016.
- Feature extraction  <a name="1.2"></a>
  - Handcrafted features
    - Robust real-time face detection, IJCV 2004.
    - Histograms of oriented gradients for human detection, CVPR 2005.
    - Integral channel features, BMVC 2009.
    - Object detection with discriminatively trained partbased models, PAMI 2010.
  - Deep features
    - Imagenet classification with deep convolutional neural networks, NIPS 2012.
    - Very deep convolutional networks for large-scale image recognition, arXiv 2014.
    - Deep residual learning for image recognition, CVPR 2016.
    - Densely connected convolutional networks, CVPR 2017.
- 1.3. Proposal classification/regression  <a name="1.3"></a>
  - Support-vector networks, ML 1995.
  - A decision-theoretic generalization of on-line learning and an application to boosting, JCSS 1997.
  - Softmax layer, Sigmoid layer, Smooth L1 layer     
- Post processing  <a name="1.4"></a>
  - Greedy NMS
  - Soft-nms–improving object detection with one line of code., ICCV 2017.
  - Learning nonmaximum suppression, CVPR 2017.
  - Relation networks for object detection, CVPR 2018.
  - Learning to separate: Detecting heavily-occluded objects in urban scenes, arXiv 2019.
  - Adaptive nms: Refining pedestrian detection in a crowd, CVPR 2020.

   
## 2. Single-spectral pedestrian detection <a name="2"></a>

#### 2.1. Handcrafted features based pedestrian detection <a name="2.1"></a>

- Decision forests based methods
  - Robust real-time face detection, IJCV 2004.
  - Integral channel features, BMVC 2009.
  - Seeking the strongest rigid detector, CVPR 2013.
  - Pedestrian detection inspired by appearance constancy and shape symmetry, CVPR 2016/TIP 2016.
  - Semantic channels for fast pedestrian detection, CVPR 2016.
  - Fast boosting based detection using scale invariant multimodal multiresolution filtered features, CVPR 2017.
  - Fast feature pyramids for object detection, BMVC 2010/PAMI 2014.
  - Pedestrian detection by feature selected self-similarity features, IEEE Access 2018.
  - Crosstalk cascades for frame-rate pedestrian detection, ECCV 2012.
  - Local decorrelation for improved pedestrian detection, NIPS 2014.
  - Local co-occurrence selection via partial least squares for pedestrian detection, TITS 2019.
  - Learning sampling distributions for efficient object detection, TIP 2017.
  - Exploring weak stabilization for motion feature extraction, CVPR 2013.
  - Looking at pedestrians at different scales: A multiresolution approach and evaluations, TITS 2016.
  - Pedestrian detection with spatially pooled features and structured ensemble learning, ECCV 2016/PAMI 2017.
  - Lbp channels for pedestrian detection, WACV 2018.
  - New features and insights for pedestrian detection, CVPR 2010.
  - A novel pixel neighborhood differential statistic feature for pedestrian and face detection, PR 2017.
  - Pedestrian proposal and refining based on the shared pixel differential feature, TITS 2019.
  - An extended filtered channel framework for pedestrian detection, TITS 2018.
  - Informed haar-like features improve pedestrian detection, CVPR 2014.
  - Exploring human vision driven features for pedestrian detection, TCSVT 2015.
  - How far are we from solving pedestrian detection? CVPR 2016
  - Filtered channel features for pedestrian detection, CVPR 2015.
  - Group cost-sensitive boostlr with vector form decorrelated filters for pedestrian detection, TITS 2019.
  - Discriminative latent semantic feature learning for pedestrian detection, Neurocomputing 2017.
- Deformable part based methods
  - Object detection with discriminatively trained partbased models, TPAMI 2010.
  - Histograms of oriented gradients for human detection, CVPR 2015.
  - A pedestrian detection system accelerated by kernelized proposals, TITS 2019.
  - Cascade object detection with deformable part models, CVPR 2010.
  - Real-time rgb-d based template matching pedestrian detection, ICRA 2016.
  - Pedestrian detection using pixel difference matrix projection, TITS 2019.
  - Single-pedestrian detection aided by multi-pedestrian detection, CVPR 2013/TPAMI 2015.
  - Multiresolution models for object detection, ECCV 2010.
  - Pedestrian detection in crowded scenes via scale and occlusion analysis, ICIP 2016.
  - Robust multi-resolution pedestrian detection in traffic scenes, CVPR 2013.
#### 2.2. Deep features based pedestrian detection <a name="2.2"></a>
- Hybrid pedestrian detection methods
  - Real-time pedestrian detection with deep network cascades, BMVC 2015.
  - Learning complexity-aware cascades for deep pedestrian detection, ICCV 2015.
  - Learning multilayer channel features for pedestrian detection, TIP 2017.
  - S-cnn: Subcategory-aware convolutional networks for object detection, TPAMI 2018.
  - Taking a deeper look at pedestrians, CVPR 2015.
  - Deep network aided by guiding network for pedestrian detection, PRL 2017.
  - Scale-aware fast r-cnn for pedestrian detection, TMM 2018.
  - Neural features for pedestrian detection, Neurocomputing 2017.
  - Switchable deep network for pedestrian detection, CVPR 2014.
  - Joint deep learning for pedestrian detection, ICCV 2013.
  - Jointly learning deep features, deformable parts, occlusion and classification for pedestrian detection, TPAMI 2018.
  - Improving the performance of pedestrian detectors using convolutional learning, PR 2017.
  - Filtered shallow-deep feature channels for pedestrian detection, Neurocomputing 2017.
  - Deep learning strong parts for pedestrian detection, ICCV 2015.
  - Hybrid channel based pedestrian detection, Neurocomputing 2017.
  - Pushing the limits of deep cnns for pedestrian detection, TCSVT 2018.
  - Fast pedestrian detection with attention-enhanced multi-scale rpn and soft-cascaded decision trees, TITS 2019.
  - Convolutional channel features, ICCV 2015.
  - Is faster r-cnn doing well for pedestrian detection? ECCV 2016.
-  Pure CNN based pedestrian detection methods
  - Scale-aware methods
    - Fpn++: A simple baseline for pedestrian detection, ICME 2019.
    - A unified multi-scale deep convolutional neural network for fast object detection, ECCV 2016.
    - Sam-rcnn: Scaleaware multi-resolution multi-channel pedestrian detection, BMVC 2018.
    - Exploit all the layers: Fast and accurate cnn object detector with scale dependent pooling and cascaded rejection classifiers, CVPR 2016.
    - Scale-adaptive deconvolutional regression network for pedestrian detection, ACCV 2016.
  - Part-based methods
    - Pedjointnet: Joint headshoulder and full body deep network for pedestrian detection, IEEE Access 2019.
    - Semantic head enhanced pedestrian detection in a crowd, arXiv 2019.
    - Deepid-net: Object detection with deformable part based convolutional neural networks, TPAMI 2017.
    - Mask-guided attention network for occluded pedestrian detection, ICCV 2019.
    - Semantic part rcnn for real-world pedestrian detection, CVPRW 2019.
    - Double anchor r-cnn for human detection in a crowd, arXiv 2019.
    - Occlusion-aware r-cnn: Detecting pedestrians in a crowd, ECCV 2018.
    - Joint holistic and partial cnn for pedestrian detection, BMVC 2018.
    - Bi-box regression for pedestrian detection and occlusion estimation, ECCV 2018.
  - Attention-based methods
    - Illuminating pedestrians via simultaneous detection and segmentation, ICCV 2017.
    - Deep feature fusion by competitive attention for pedestrian detection, IEEE Access 2019.
    - Vis-hud: Using visual saliency to improve human detection with convolutional neural networks, CVPRW 2018.
    - Graininessaware deep feature learning for pedestrian detection, ECCV 2018.
    - Multi-grained deep feature learning for robust pedestrian detection, TCSVT 2019.
    - Part-level convolutional neural networks for pedestrian detection using saliency and boundary box alignment, IEEE Access 2019.
    - Occluded pedestrian detection through guided attention in cnns, CVPR 2018.
    - Attention guided neural network models for occluded pedestrian detection, PR 2020.
  - Feature-fused methods
    - Learning pixel-level and instance-level context-aware features for pedestrian detection in crowds, IEEE Access 2019.
    - Object detection based on multilayer convolution feature fusion and online hard example mining, IEEE Access 2018.
    - Direct multi-scale dual-stream network for pedestrian detection, ICIP 2017.
    - Accurate single stage detector using recurrent rolling convolution, CVPR 2017.
    - Coupled network for robust pedestrian detection with gated multi-layer feature extraction and deformable occlusion handling Cascade-based methods, arXiv 2019.
    - Deep aggregation learning for high-performance small pedestrian detection, ACML 2018.
    - Pedestrian detection via body part semantic and contextual information with dnn, TMM 2018.
    - Temporal-context enhanced detection of heavily occluded pedestrians, CVPR 2020.
    - Object detection with location-aware deformable convolution and backward attention filtering, CVPR 2019.
    - Mfr-cnn: Incorporating multi-scale features and global information for traffic object detection, TVT 2019.
  - Cascade-based methods
    - Fused dnn: A deep neural network fusion approach to fast and robust pedestrian detection, WACV 2017.
    - Pedestrian detection: The elephant in the room, arXiv 2020.
    - Pedestrian detection with autoregressive network phases, CVPR 2019.
    - Learning efficient single-stage pedestrian detectors by asymptotic localization fitting, ECCV 2018.
    - A one-and-half stage pedestrian detector, WACV 2020.
    - Circlenet: Reciprocating feature adaptation for robust pedestrian detection, TITS 2019.
  - Anchor-free methods
    - High-level semantic feature detection: A new perspective for pedestrian detection, CVPR 2019.
    - Small-scale pedestrian detection based on topological line localization and temporal feature aggregation, ECCV 2018.
    - Attribute-aware pedestrian detection in a crowd, arXiv 2019.
  - Data-augmentation methods
    - A shape transformation-based dataset augmentation framework for pedestrian detection, arXiv 2019.
    - Pedhunter: Occlusion robust pedestrian detector in crowded scenes, AAAI 2020.
    - Synthesizing a scene-specific pedestrian detector and pose estimator for static video surveillance, IJCV 2018.
    - Where, what, whether: Multi-modal learning meets pedestrian detection, CVPR 2020.
    - Advanced pedestrian dataset augmentation for autonomous driving, ICCVW 2019.
    - Pmc-gans: Generating multi-scale high-quality pedestrian with multimodal cascaded gans, BMVC 2019.
    - Training cascade compact cnn with region-iou for accurate pedestrian detection, TITS 2019.
  - Loss-driven methods
    - Learning lightweight pedestrian detector with hierarchical knowledge distillation, ICIP 2019.
    - Perceptual generative adversarial networks for small object detection, CVPR 2017.
    - Mimicking very efficient network for object detection, CVPR 2017.
    - Fused discriminative metric learning for low resolution pedestrian detection, ICIP 2017.
    - Repulsion loss: Detecting pedestrians in a crowd, CVPR 2018.
    - Boosted convolutional neural networks (bcnn) for pedestrian detection, WACV 2017.
    - Subcategory-aware convolutional neural networks for object proposals and detection, WACV 2017.
    - Discriminative feature transformation for occluded pedestrian detection, ICCV 2019.
  - Post-processing methods
    - Nms by representative region: Towards crowded pedestrian detection by proposal pairing, CVPR 2020.
    - Adaptive nms: Refining pedestrian detection in a crowd, CVPR 2019.
    - End-toend people detection in crowded scenes, CVPR 2016.
    - S3d: Scalable pedestrian detection via score scale surface discrimination, TCSVT 2020.
    - Learning to separate: Detecting heavily-occluded objects in urban scenes, arXiv 2019.
    - Single shot multibox detector with kalman filter for online pedestrian detection in video, IEEE Access 2019.
    - Led: Localization-quality estimation embedded detector, ICIP 2018.
  - Multi-task methods
    - Re-id driven localization refinement for person search, ICCV 2019.
    - Cluenet: A deep framework for occluded pedestrian pose estimation, BMVC 2019.
    - What can help pedestrian detection? CVPR 2017.
    - Human detection aided by deeply learned semantic masks, TCSVT 2019.
    - Semantic part rcnn for real-world pedestrian detection, CVPRW 2019.
    - Accurate pedestrian detection by human pose regression, TIP 2019.
  - Pedestrian detection on thermal or fish-eye images
    - Pedestrian detection from thermal images using saliency maps, CVPRW 2019.
    - Domainadaptive pedestrian detection in thermal images, ICIP 2019.
    - Deep learning approaches on pedestrian detection in hazy weather, TIE 2019.
    - Spatial focal loss for pedestrian detection in fisheye imagery, WACV 2019.
    - Oriented spatial transformer network for pedestrian detection using fish-eye camera, TMM 2020.
    - Exploiting target data to learn deep convolutional networks for scene-adapted human detection, TIP 2018.
    - Semi-supervised human detection via region proposal networks aided by verification, TIP 2020.

## 3. Multispectral pedestrian detection <a name="3"></a>

#### 3.1. Deep feature fusion <a name="3.1"></a>

  - Pedestrian detection for autonomous vehicle using multi-spectral cameras, TIV 2019.
  - Fusion of multispectral data through illuminationaware deep neural networks for pedestrian detection, IF 2019.
  - Fully convolutional region proposal networks for multispectral person detection, CVPR 2017.
  - Illuminationaware faster r-cnn for robust multispectral pedestrian detection, PR 2019.
  - Multispectral deep neural networks for pedestrian detection, BMVC 2016.
  - Cross-modality interactive attention network for multispectral pedestrian detection, IF 2019.
  
#### 3.1. Data processing <a name="3.1"></a>
  - Weakly aligned cross-modal learning for multispectral pedestrian detection, ICCV 2019.
  - Multispectral pedestrian detection via simultaneous detection and segmentation, BMVC 2018.

#### 3.1. Domain adaptation <a name="3.1"></a>
  - Pedestrian detection with unsupervised multispectral feature learning using deep neural networks, IF 2019.
  - Unsupervised domain adaptation for multispectral pedestrian detection, CVPRW 2019.
  -  Learning cross-modal deep representations for robust pedestrian detection, CVPR 2017.
  
## 4. Datasets <a name="4"></a>

#### 4.1. Earlier pedestrian datasets <a name="4.1"></a>
   - Monocular pedestrian detection: Survey and experiments, TPAMI 2019.
   - A trainable system for object detection, IJCV 2000.
   - Histograms of oriented gradients for human detection, CVPR 2005.
   - Depth and appearance for mobile scene analysis, ICCV 2007.
   - Multi-cue onboard pedestrian detection, CVPR 2009.

#### 4.1. Modern pedestrian datasets <a name="4.2"></a>
   - Pedestrian detection: An evaluation of the state of the art, PAMI 2010.
   - Are we ready for autonomous driving? the kitti vision benchmark suite, CVPR 2012
   - Eurocity persons: A novel benchmark for person detection in traffic scenes, PAMI 2019.
   - Nightowls: A pedestrians at night dataset, ACCV 2018.
   - Crowdhuman: A benchmark for detecting human in a crowd, arXiv 2018.
   - Citypersons: A diverse dataset for pedestrian detection, CVPR 2017.
   - Widerperson: A diverse dataset for dense pedestrian detection in the wild, TMM 2020.

#### 4.3. Multispectral pedestrian datasets <a name="4.3"></a>
   - Multispectral pedestrian detection: Benchmark dataset and baseline, CVPR 2015.
   - Pedestrian detection at day/night time with visible and fir cameras: A comparison, PR 2016.

## 5. Challenges <a name="5"></a>

#### 5.1. Scale variance <a name="5.1"></a>
   - A unified multi-scale deep convolutional neural network for fast object detection, ECCV 2016.
   - High-level semantic networks for multi-scale object detection, TCSVT 2019.
   - Small-scale pedestrian detection based on deep neural network, TITS 2019.
   - Scale-aware fast r-cnn for pedestrian detection, TMM 2019.
   - Exploit all the layers: Fast and accurate cnn object detector with scale dependent pooling and cascaded rejection classifiers, CVPR 2016.
   - Feature pyramid networks for object detection, CVPR 2017.
   - Perceptual generative adversarial networks for small object detection, CVPR 2017.
   - Jcs-net: Joint classification and super-resolution network for small-scale pedestrian detection in surveillance images, TIFS 2019.
   - Task-driven super resolution: Object detection in low-resolution images, arXiv 2018.
   - Multi-resolution generative adversarial networks for tinyscale pedestrian detection, ICIP 2019.
   - Scale match for tiny person detection, WACV 2020.
   
#### 5.2. Occlusion <a name="5.2"></a>
   - Detection in crowded scenes: One proposal, multiple predictions, CVPR 2020.
   - Pedhunter: Occlusion robust pedestrian detector in crowded scenes, AAAI 2020.
   - Relational learning for joint head and human detection, AAAI 2020.
   - Adaptive nms: Refining pedestrian detection in a crowd, CVPR 2019.
   - Mask-guided attention network for occluded pedestrian detection, ICCV 2019.
   - Improving occlusion and hard negative handlingfor single-stage pedestrian detectors, CVPR 2018.
   - Handling occlusions with franken-classifiers, CVPR 2013.
   - Deep learning strong parts for pedestrian detection, ICCV 2015.
   - Repulsion loss: Detecting pedestrians in a crowd, CVPR 2018.
   - An hog-lbp human detector with partial occlusion handling, ICCV 2010.
   - Psc-net: Learning part spatial cooccurence for occluded pedestrian detection, arXiv 2010.
   - Learning to separate: Detecting heavily-occluded objects in urban scenes, arXiv 2019.
   - Occlusion-aware r-cnn: Detecting pedestrians in a crowd, ECCV 2018.
   - Multi-label learning of part detectors for heavily occluded pedestrian detection, ICCV 2017.
   - Bi-box regression for pedestrian detection and occlusion estimation, ECCV 2018.

#### 5.3. Domain adaptation <a name="5.3"></a>
   - Domain adaptive faster r-cnn for object detection in the wild, CVPR 2018.
   - Progressive domain adaptation for object detection, CVPRW 2018.
   - A robust learning approach to domain adaptive object detection, ICCV 2019.
   - Diversify and match: A domain adaptive representation learning paradigm for object detection, CVPR 2019.
   - Domain adaptation for object detection via style consistency, BMVC 2019.
   - Strong-weak distribution alignment for adaptive object detection, CVPR 2019.
   -  Few-shot adaptive faster r-cnn, CVPR 2019.
   - Multi-level domain adaptive learning for cross-domain detection, ICCVW 2019.
   - Adapting object detectors via selective cross-domain alignment, CVPR 2019.



## Contact 
Please contact us for your questions about this webpage.

