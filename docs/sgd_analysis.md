# Analysis: SGD Optimizer Experiment

## 1. Model Architecture
We built a **Convolutional Neural Network (CNN)**.
- **Layers**: 3 sets of `Conv2D` + `MaxPooling` layers.
- **Hidden Layer**: A `Dense(128)` layer (the "embedding" layer).
- **Output**: A final `Dense` layer with 126 neurons (one for each class).

## 2. Optimizer Choice
We chose **SGD (Stochastic Gradient Descent)**.
- This is the default or standard way models learn.
- It doesn't have any extra features like momentum (speeding up) or adaptive rates (like Adam).
- It often works well as a baseline to see how the most basic optimizer performs.

## 3. Why We Used This Architecture
We kept it simple because our images are tiny (32x32 pixels).
- **3 Convolution Layers**: Enough to capture shapes without making the model massive/slow.
- **Dense(128)**: I bumped this up from 64 to 128 to give the AI more power and match the other models in the team.

## 4. Why This Optimizer Performed Well (or Poorly)
SGD is like hiking down a mountain with a consistent step size.
- **Good**: It's stable and predictable. It doesn't jump around wildly if you pick the right settings.
- **Bad**: It can be very slow, other optimizers like Adam usually learned faster. My SGD needed all 35-50 epochs to get good results.

## 5. Optimizer Results (Per Learning Rate)

| Learning Rate | Result | Notes |
| :--- | :--- | :--- |
| **0.001** | **Too Slow** | Accuracy stuck at ~15%. It was barely moving. |
| **0.05** | **Overfitting** | It learned fast (hit ~89% training accuracy) but got confused on new data (~62.5% accuracy). |
| **0.01** | **Solid Baseline** | Similar to 0.05 but a bit smoother / safer. |

## 6. My Conclusions (Why the results are this way)
I think 126 classes is surprisingly hard for a simple model, and SGD needs more time than we gave it. However, extending the training time would be a waste of time and resources. At that point, we would be better off using a different optimizer.
