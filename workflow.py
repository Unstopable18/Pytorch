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
# print(f'x -> {len(x)}\n{x[:10]}')
# print(f'y -> {len(y)}\n{y[:10]}')

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

# print(f'------ Train Test Split ------\nX_train -> {X_train}\ny_train -> {y_train}\nX_test -> {X_test}\ny_test -> {y_test}')
# print(len(X_train),len(y_train),len(X_test),len(y_test))
def plot_predictions(train_data=X_train,train_label=y_train,test_data=X_test,test_label=y_test,predictions=None):
    plt.figure(figsize=(10,7))
    plt.scatter(train_data,train_label,c='b',s=4,label='Training Data')
    plt.scatter(test_data,test_label,c='g',s=4,label='Testing Data')

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
# print(list(model_0.parameters()))
# print(model_0.state_dict())
with torch.inference_mode():  #inference mode disables grad, more faster
    y_pred=model_0(X_test)
# print(f'y_pred >>>>>>>>>>>>>\n{y_pred}')
# plot_predictions(predictions=y_pred)

loss_fn=nn.L1Loss()
optimizer=torch.optim.SGD(params=model_0.parameters(),lr=0.01)
epochs=150

epoch_count=[]
train_loss_values=[]
test_loss_values=[]

for epoch in range(epochs):
    model_0.train()
    y_pred=model_0(X_train)
    loss=loss_fn(y_pred,y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    model_0.eval()
    with torch.inference_mode():  #inference mode disables grad, more faster
        test_pred=model_0(X_test)
        test_loss=loss_fn(test_pred,y_test)
    if epoch%10==0:
        epoch_count.append(epoch)
        train_loss_values.append(loss.detach().numpy())
        test_loss_values.append(test_loss.detach().numpy())
        print(f'Epoch -> {epoch}) | Loss  -> {loss}| Test Loss  -> {test_loss}')
print(model_0.state_dict())
# plot_predictions(predictions=test_pred)

plt.plot(epoch_count, train_loss_values, label="Train loss")
plt.plot(epoch_count, test_loss_values, label="Test loss")
plt.title("Training and test loss curves")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()
# mpld3.show()
torch.save(model_0.state_dict(),'workflow_model_1.pth')
print('Model Saved .....')
model_1=LinearRegressionModel()
model_1.load_state_dict(torch.load('workflow_model_1.pth'))
print('Loaded Model ----> ',model_1.state_dict())