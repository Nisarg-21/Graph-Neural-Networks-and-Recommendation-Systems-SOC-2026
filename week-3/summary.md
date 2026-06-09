# q3

### Theory: Prove the existence of an MLP (including its weights) that can perfectly learn the identity function 

the given function is a linear function.a single layer with one weight and one bias can represent this linear function without any activation function.
output= W*X + B 

### Is an activation function needed here? Why or why not?

no,activation function is not needed for linear functions.

### What do you expect the exact weights and biases to be?

expected weight to be 1 and bias to be zero

### Defend your design—simpler architectures with fewer layers are strongly preferred.

no extra layers needed as there is no complexity or non-linearity to learn.and adding linear equations is mathematically equivalent to a single linear equations.

### Are the weights exactly what you expected? If not, explain why.

final weights : W = 1.000000 , b = -0.000040 
because at the end the gradient becomes very small and thus the value of b stops getting updated furhter.

# q4

### Theory: Prove the existence of an MLP (including its weights) that can perfectly learn the function

dont know exactly how to prove theoretically, but universal approximation theorem states:
An MLP with at least one hidden layer and a non-linear activation can approximate any continuous function to arbitrary precision, given enough neurons.

### Is an activation function needed?

yes,as the function is non-linear.

### What are the exact expected weights?
there will be no exact patterns for non linear functions

# q5

### When training a neural network for classification tasks, which loss function typically performs better: Mean Squared Error (MSE) with tanh or Cross-Entropy with Softmax?

Cross-Entropy with Softmax performs better for classification tasks. Cross-entropy is specifically designed to measure the difference between predicted probability distributions and true labels, making it naturally suited for classification. Softmax ensures outputs are valid probabilities.

MSE with tanh was designed for regression. Using it for classification produces weaker gradients,suffers from vanishing gradient problems due to tanh's saturating nature, and doesn't penalize wrong class probabilities as effectively as cross-entropy does.