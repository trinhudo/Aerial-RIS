# Aerial Reconfigurable Intelligent Surface-Aided Wireless Communication Systems

**Tri Nhu Do, Georges Kaddoum, Thanh Luan Nguyen, Daniel Benevides da Costa, and Zygmunt J. Haas**  
_IEEE International Symposium on Personal, Indoor and Mobile Radio Communications (PIMRC), 2021_

## Abstract
In this paper, we propose and investigate an aerial reconfigurable intelligent surface (aerial-RIS)-aided wireless com- munication system. Specifically, considering practical composite fading channels, we characterize the air-to-ground (A2G) links by Namkagami-m small-scale fading and inverse-Gamma large- scale shadowing. To investigate the delay-limited performance of the proposed system, we derive a tight approximate closed- form expression for the end-to-end outage probability (OP). Next, considering a mobile environment, where performance analysis is intractable, we rely on machine learning-based performance prediction to evaluate the performance of the mobile aerial- RIS-aided system. Specifically, taking into account the three- dimensional (3D) spatial movement of the aerial-RIS, we build a deep neural network (DNN) to accurately predict the OP. We show that: (i) fading and shadowing conditions have strong impact on the OP, (ii) as the number of reflecting elements increases, aerial-RIS achieves higher energy efficiency (EE), and (iii) the aerial-RIS-aided system outperforms conventional relaying systems.

## Paper
- [View pdf](https://github.com/trinhudo/AerialRIS/blob/main/Aerial_RIS_manuscript.pdf)

## Source code
### Outage performance analysis
- [Demo by MATLAB](https://github.com/trinhudo/AerialRIS/blob/main/demo_OP_ana_sim.pdf)
- [Source code](https://github.com/trinhudo/Aerial-RIS/tree/main/Outage_Analysis)

### Outage probability prediction using deep neural network (DNN)
- [Demo by Google Colab](https://github.com/trinhudo/AerialRIS/blob/main/Aerial_RIS_DNN_OP_prediction_Colaboratory.pdf)
- [ipynb](https://github.com/trinhudo/AerialRIS/blob/main/DNN_prediction/Aerial_RIS_DNN_OP_prediction.ipynb)
- [Source code](https://github.com/trinhudo/Aerial-RIS/tree/main/DNN_prediction)
