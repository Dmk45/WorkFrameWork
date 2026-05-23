How framework should work(uability wise)



Define NN(coudl be part of parent class)
add layers to nn
for epochs
run forward pass
run backward pass
update weights
print metrics


forward pass psuedo code:
- for each layer in nn:
    - call layer.forward with input
    - store output
- return final output


backward psuedo code:
- for each layer in nn (in reverse order):
    - call layer.backward with input
    - store gradient
- return final gradient
clear cache/zero grad per batch