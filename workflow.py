import torch
from torch import nn # nn contains all of PyTorch's building blocks for neural networks
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import mpld3


# Check PyTorch version
print(torch.__version__)

weight=0.7
bias=0.3
start=0
end=1
step=0.02
x=torch.arange(start,end,step).unsqueeze(dim=1)
y=weight*x+bias
print(f'x -> {len(x)}\n{x[:10]}')
print(f'y -> {len(y)}\n{y[:10]}')

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

print(f'------ Train Test Split ------\nX_train -> {X_train}\ny_train -> {y_train}\nX_test -> {X_test}\ny_test -> {y_test}')
print(len(X_train),len(y_train),len(X_test),len(y_test))
def plot_predictions(train_data=X_train,train_label=y_train,test_data=X_test,test_label=y_test,predictions=None):
    plt.figure(figsize=(10,7))
    plt.scatter(train_data,train_label,c='b',s=4,label='Training Data')
    plt.scatter(test_data,test_label,c='b',s=4,label='Testing Data')

    if predictions is not None:
        plt.scatter(test_data,predictions,c='r',s=4,label='Predictions')
    plt.legend(prop={'size':14})
    mpld3.show()

# plot_predictions()

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights=nn.Parameter(torch.randn(1,requires_grad=True,dtype=float))
        self.bias=nn.Parameter(torch.randn(1,requires_grad=True,dtype=float))

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        return self.weights*x+self.bias

torch.manual_seed(42)
model_0=LinearRegressionModel()
print(list(model_0.parameters()))
print(model_0.state_dict())
with torch.inference_mode():
    y_pred=model_0(X_test)
print(f'y_pred >>>>>>>>>>>>>\n{y_pred}')
plot_predictions(predictions=y_pred)