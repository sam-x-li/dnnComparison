# dnnComparison: Deep Feedforward Network vs K-Nearest Neighbours on MNIST

This project was part of my Extended Project Qualification (EPQ) exploring the performance of neural networks against traditional machine learning models, using the medium of handwritten digit recognition. 
Find the paper here: https://drive.google.com/file/d/1zSHU0O1X7Z1dGi1xpU2J_2SEH-dbQnRz/view?usp=sharing

mnistloader.py is used to convert MNIST bitmaps into the correct form. This was taken from Michael Nielsen's github repository: https://github.com/mnielsen/neural-networks-and-deep-learning

--- 

## Summary
- Made a deep feedforward network from scratch and compared against k-nearest neighbours (kNN) on MNIST
- Compared time taken and accuracy while varying the size of the training set
- Found that kNN consistently had higher accuracy (p < 0.05, paired t-test)
- KNN was faster for smaller training set sizes (n <= 150), while the dNN was faster for n > 150.
- Supported with a literature review

---

## Sample Results

```
Training set size (n): 50 Accuracy: 0.5778 Time taken/s: 0.5811195373535156 Testing set size: 10000 Number of successes: 5778
Training set size (n): 150 Accuracy: 0.652 Time taken/s: 2.432004928588867 Testing set size: 10000 Number of successes: 6520
Training set size (n): 500 Accuracy: 0.7749 Time taken/s: 16.899497270584106 Testing set size: 10000 Number of successes: 7749
Training set size (n): 1000 Accuracy: 0.7804 Time taken/s: 36.08581876754761 Testing set size: 10000 Number of successes: 7804
```

---

## Images

<img width="1093" height="657" alt="Mean Accuracy Bar" src="https://github.com/user-attachments/assets/760a6483-5a2a-45f2-9f91-e8d68a222def" />
<img width="948" height="707" alt="Percentage Accuracy Scatter" src="https://github.com/user-attachments/assets/90580add-0202-4e96-9c54-7c1f347342e9" />
<img width="987" height="675" alt="Mean Time Bar" src="https://github.com/user-attachments/assets/029daedf-8948-4c86-ab6d-0c1b0a7b6bd4" />
<img width="950" height="628" alt="Time Scatter A" src="https://github.com/user-attachments/assets/b789c158-7034-4f1a-9560-3858bc9bf673" />
<img width="902" height="701" alt="Time Scatter B" src="https://github.com/user-attachments/assets/3f1715bc-052e-433d-8a1d-7e56a057fc9e" />
