# IRP_SS_Fusion_Spirent
## IRP about Sensor fusion using Transformer NN
A first use case of using Transformer architecture to design a sensor fusion (IMU/GNSS) algorithm for pose estimation. The dataset has been generated in Hardware in the loop simulation with Spirent's simulator to get the GNSS data, that are then process to a consummer-grade Ublox gnss receiver with SIMSensors to generate the corresponding IMU data.  
Please see the PDF file to have an insight of the main methodology of the project.  
The vel_pred_INS_navigation.py file contains the code used to predict the velocities and corresponding covariance of the prediction based one IMU measurements only. The pre processing part of the data is not implemented here.  We followed **CTIN: Robust Contextual Transformer Network for Inertial Navigation** for the implementation. We modified the ResNet used, as well as the regression (MLP).   
Next steps are to add with the same idea other measurements such as GNSS ones.
