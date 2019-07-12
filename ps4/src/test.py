def forward_softmax(x):
    """
    Compute softmax function for a single example.
    The shape of the input is of size # num classes.

    Important Note: You must be careful to avoid overflow for this function. Functions
    like softmax have a tendency to overflow when very large numbers like e^10000 are computed.
    You will know that your function is overflow resistent when it can handle input like:
    np.array([[10000, 10010, 10]]) without issues.

        x: A 1d numpy float array of shape number_of_classes

    Returns:
        A 1d numpy float array containing the softmax results of shape  number_of_classes
    """
    x = x - np.max(x,axis=0)
    exp = np.exp(x)
    s = exp / np.sum(exp,axis=0)
    return s
def backward_softmax_my(x, grad_outputs):
    """
    Compute the gradient of the loss with respect to x.

    grad_outputs is the gradient of the loss with respect to the outputs of the softmax.

    Args:
        x: A 1d numpy float array of shape number_of_classes
        grad_outputs: A 1d numpy float array of shape number_of_classes

    Returns:
        A 1d numpy float array of the same shape as x with the derivative of the loss with respect to x
    """
    
    # *** START CODE HERE ***
    y = forward_softmax(x)
    dy_dx = np.diag(y) - np.outer(y, y)
    dL_dx = dy_dx.dot(grad_outputs)
    
    return dL_dx

def backward_softmax(x, grad_outputs):
    """
    Compute the gradient of the loss with respect to x.
    grad_outputs is the gradient of the loss with respect to the outputs of the softmax.
    Args:
        x: A 1d numpy float array of shape number_of_classes
        grad_outputs: A 1d numpy flaot array of shape number_of_classes
    Returns:
        A 1d numpy float array of the same shape as x with the derivative of the loss with respect to x
    """

    # *** START CODE HERE ***
    n = x.size
    grads = np.zeros((n, n))
    exp = np.exp(x)
    denominator = np.sum(exp)
    for i in range(n):
        for j in range(n):
            grads[i][j] = np.exp(x[i]) * (denominator - np.exp(x[i])) if i == j else -1 * np.exp(x[i]) * np.exp(x[j])

    return np.matmul(grad_outputs, grads / np.square(denominator))
    # *** END CODE HERE ***

def rel_error(x, y):
    return np.abs(x - y).max() / np.abs(x).max()

def backward_convolution(conv_W, conv_b, data, output_grad):
    """
    Compute the gradient of the loss with respect to the parameters of the convolution.
    See forward_convolution for the sizes of the arguments.
    output_grad is the gradient of the loss with respect to the output of the convolution.
    Returns:
        A tuple containing 3 gradients.
        The first element is the gradient of the loss with respect to the convolution weights
        The second element is the gradient of the loss with respect to the convolution bias
        The third element is the gradient of the loss with respect to the input data
    """

    # *** START CODE HERE ***
    conv_channels, _, conv_width, conv_height = conv_W.shape

    input_channels, input_width, input_height = data.shape

    # output = np.zeros((conv_channels, input_width - conv_width + 1, input_height - conv_height + 1))
    d_conv = np.zeros((conv_channels, input_channels, conv_width, conv_height))
    d_bias = np.zeros((conv_channels))
    d_data = np.zeros((input_channels, input_width, input_height))

    for x in range(input_width - conv_width + 1):
        for y in range(input_height - conv_height + 1):
            for output_channel in range(conv_channels):
                d_bias[output_channel] += output_grad[output_channel, x, y]
                for di in range(conv_width):
                    for dj in range(conv_height):
                        for input_channel in range(input_channels):
                            d_conv[output_channel, input_channel, di, dj] += output_grad[output_channel, x, y] * data[input_channel, x + di, y + dj]
                            d_data[input_channel, x + di, y + dj] += output_grad[output_channel, x, y] * conv_W[output_channel, input_channel, di, dj]

    return d_conv, d_bias, d_data

def backward_convolution_my(conv_W, conv_b, data, output_grad):
    """
    Compute the gradient of the loss with respect to the parameters of the convolution.

    See forward_convolution for the sizes of the arguments.
    output_grad is the gradient of the loss with respect to the output of the convolution.

    Returns:
        A tuple containing 3 gradients.
        The first element is the gradient of the loss with respect to the convolution weights
        The second element is the gradient of the loss with respect to the convolution bias
        The third element is the gradient of the loss with respect to the input data
    """

    # *** START CODE HERE ***
    dconv_W = np.zeros_like(conv_W)
    dconv_b = np.zeros_like(conv_b)
    ddata = np.zeros_like(data)
    
    C_out, C_in, W_conv, H_conv = conv_W.shape
    _, W_in, H_in = data.shape
    _, W_out, H_out = output_grad.shape
    
    # iterate through each output unit
    for c in range(C_out):
        for w in range(W_out):
            for h in range(H_out):
                dconv_W[c] += data[:, w:w+W_conv, h:h+H_conv] * output_grad[c, w, h]
                ddata[:, w:w+W_conv, h:h+H_conv] += conv_W[c] * output_grad[c, w, h]
                dconv_b[c] += output_grad[c, w, h]
    return dconv_W, dconv_b, ddata

def backward_linear_my(weights, bias, data, output_grad):
    """
    Compute the gradients of the loss with respect to the parameters of a linear layer.

    See forward_linear for information about the shapes of the variables.

    output_grad is the gradient of the loss with respect to the output of this layer.

    This should return a tuple with three elements:
    - The gradient of the loss with respect to the weights
    - The gradient of the loss with respect to the bias
    - The gradient of the loss with respect to the data
    """

    # *** START CODE HERE ***
    ddata = output_grad.dot(weights.T)
    dweights = np.outer(data, output_grad)
    dbias = output_grad

    return dweights, dbias, ddata
    # *** END CODE HERE ***

def backward_linear(weights, bias, data, output_grad):
    """
    Compute the gradients of the loss with respect to the parameters of a linear layer.
    See forward_linear for information about the shapes of the variables.
    output_grad is the gradient of the loss with respect to the output of this layer.
    This should return a tuple with three elements:
    - The gradient of the loss with respect to the weights
    - The gradient of the loss with respect to the bias
    - The gradient of the loss with respect to the data
    """

    # *** START CODE HERE ***
    input, output = weights.shape

    d_weights = np.zeros((input, output))
    d_bias = np.zeros((output))
    d_data = np.zeros((input))

    for o in range(output):
        d_bias[o] += output_grad[o]
        for i in range(input):
            d_weights[i, o] += output_grad[o] * data[i]
            d_data[i] += output_grad[o] * weights[i, o]

    return d_weights, d_bias, d_data
    # *** END CODE HERE ***
def backward_max_pool(data, pool_width, pool_height, output_grad):
    """
    Compute the gradient of the loss with respect to the data in the max pooling layer.
    data is of the shape (# channels, width, height)
    output_grad is of shape (# channels, width // pool_width, height // pool_height)
    output_grad is the gradient of the loss with respect to the output of the backward max
    pool layer.
    Returns:
        The gradient of the loss with respect to the data (of same shape as data)
    """

    # *** START CODE HERE ***
    input_channels, input_width, input_height = data.shape
    _, output_width, output_height = output_grad.shape

    d_data = np.zeros((input_channels, input_width, input_height))

    for i in range(input_channels):
        for j in range(pool_width):
            for k in range(pool_height):
                pooled_data = data[i, j * pool_width : (j+1) * pool_width, k * pool_height : (k+1) * pool_height]
                if pooled_data.size:
                    max_elem_index_j, max_elem_index_k = np.unravel_index(pooled_data.argmax(), pooled_data.shape)
                    d_data[i, j * pool_width + max_elem_index_j, k * pool_height + max_elem_index_k] += output_grad[i, j, k]

    return d_data
    # *** END CODE HERE ***
def backward_max_pool_my(data, pool_width, pool_height, output_grad):
    """
    Compute the gradient of the loss with respect to the data in the max pooling layer.

    data is of the shape (# channels, width, height)
    output_grad is of shape (# channels, width // pool_width, height // pool_height)

    output_grad is the gradient of the loss with respect to the output of the backward max
    pool layer.

    Returns:
        The gradient of the loss with respect to the data (of same shape as data)
    """

    # *** START CODE HERE ***
    ddata = np.zeros_like(data)
    C, W_out, H_out = output_grad.shape

    # print(ddata.shape)
    # print(output_grad.shape)

    # iterate through each output unit
    for c in range(C):
        for w in range(W_out):
            for h in range(H_out):
                w_in = w * pool_width
                h_in = h * pool_height
                # print(w_in, h_in)
                # find the maximium element index
                index = np.argmax(data[c, w_in:w_in+pool_width, h_in:h_in+pool_height])
                # propagate gradient
                ddata[c, w_in:w_in+pool_width, h_in:h_in+pool_height].flat[index] += output_grad[c, w, h]
    return ddata

if __name__ == '__main__':
    import numpy as np
    a = np.random.randn(100)
    b = np.random.randn(100)
    one = backward_softmax_my(a, b)
    two = backward_softmax(a, b)
    # conv_W, conv_b, data, output_grad
    C_in = 6

    data = np.random.randn(C_in, 30, 30)
    W = H = 5
    output_grad = np.random.randn(C_in, 6, 6)
    one = backward_max_pool(data, W, H, output_grad)
    two = backward_max_pool_my(data, W, H, output_grad)
    print(rel_error(one, two))

