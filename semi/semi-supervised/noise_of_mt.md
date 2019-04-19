# stochastic augmentation
in the paper of $\Pi$ model, the added noise to the input is called stochastic 

it is not noly thing as normal noise


github:[[what is the difference between input and ema_input #17]](https://github.com/CuriousAI/mean-teacher/issues/17)
>The algorithm works by having two models that take a similar input and produce a similar output: the student model, which is a normal convolutional neural network, and a teacher model, which is the same as student except its weights are exponential moving average (EMA) of the student network. Before feeding the input to the networks, the algorithm adds noise to the input. This is done for the networks separately, sampling noise twice from the same distribution.

>In the Pytorch code, the sampling of noise twice is handled by TransformTwice class (for example [here](https://github.com/CuriousAI/mean-teacher/blob/master/pytorch/mean_teacher/datasets.py#L18-L25)). The resulting transformation takes one input image and returns two noisy versions of it. The transformation is wrapped in to the train_loader, and when the train_loader is iterated ([here](https://github.com/CuriousAI/mean-teacher/blob/master/pytorch/main.py#L215)), it returns those two noisy input images: input and ema_input. The ema_ prefix there just means that it is the one to be fed to the ema_model, i.e. the teacher network.

```python
    train_transformation = data.TransformTwice(transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ]))
```
# noise