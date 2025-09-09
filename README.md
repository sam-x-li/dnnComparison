# dnnComparison: Deep Feedforward Network vs K-Nearest Neighbours on MNIST

This project was part of my Extended Project Qualification (EPQ) exploring the performance of neural networks against traditional machine learning models, using the medium of handwritten digit recognition. 

--- 

## Summary
- Made a deep feedforward network from scratch and compared against k-nearest neighbours (kNN) on MNIST
- Compared time taken and accuracy while varying the size of the training set
- Found that kNN consistently had higher accuracy (p < 0.05, paired t-test)
- KNN was faster for smaller training set sizes (n <= 150), while the dNN was faster for n > 150.

## Images

<img width="1093" height="657" alt="Mean Accuracy Bar" src="https://github.com/user-attachments/assets/760a6483-5a2a-45f2-9f91-e8d68a222def" />
<img width="948" height="707" alt="Percentage Accuracy Scatter" src="https://github.com/user-attachments/assets/90580add-0202-4e96-9c54-7c1f347342e9" />
<img width="987" height="675" alt="Mean Time Bar" src="https://github.com/user-attachments/assets/029daedf-8948-4c86-ab6d-0c1b0a7b6bd4" />
<img width="950" height="628" alt="Time Scatter A" src="https://github.com/user-attachments/assets/b789c158-7034-4f1a-9560-3858bc9bf673" />
<img width="902" height="701" alt="Time Scatter B" src="https://github.com/user-attachments/assets/3f1715bc-052e-433d-8a1d-7e56a057fc9e" />
