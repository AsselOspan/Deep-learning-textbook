"""Проверка обратного прохода свёртки с заданной матрицей производных G."""
import torch
from torch.nn.functional import conv2d

from convolution_numeric import example


def verify_gradients():
    image_values, kernel_values = example()
    image = torch.as_tensor(image_values, dtype=torch.float64).clone().requires_grad_()
    kernel = torch.as_tensor(kernel_values, dtype=torch.float64).clone().requires_grad_()
    bias = torch.zeros(1, dtype=image.dtype, requires_grad=True)
    g = torch.tensor([[2, -1, 0], [3, 0, 0], [0, 0, 0]], dtype=image.dtype)
    output = conv2d(image[None, None], kernel[None, None], bias)[0, 0]
    loss = (output * g).sum()
    loss.backward()

    manual_kernel = torch.zeros_like(kernel)
    manual_image = torch.zeros_like(image)
    kh, kw = kernel.shape
    for i in range(g.shape[0]):
        for j in range(g.shape[1]):
            rows, cols = slice(i, i + kh), slice(j, j + kw)
            manual_kernel += g[i, j] * image.detach()[rows, cols]
            manual_image[rows, cols] += g[i, j] * kernel.detach()
    manual_bias = g.sum().reshape_as(bias)

    for actual, expected in ((kernel.grad, manual_kernel), (bias.grad, manual_bias),
                             (image.grad, manual_image)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    return manual_kernel, manual_bias, manual_image


if __name__ == '__main__':
    kernel, bias, image = verify_gradients()
    print('Градиент ядра:', kernel.tolist())
    print('Градиент смещения:', bias.item())
    print('Градиент входа:', image.tolist())
