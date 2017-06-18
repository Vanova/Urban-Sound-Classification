"""
TODO: finish implementation
"""
from recurrentshop import LSTMCell, RecurrentContainer


def attentionLayer(input_layer, output_dim, output_length, hidden_dim=None, depth=1, bidirectional=True, dropout=0., **kwargs):
    """
    This is an attention Seq2seq model based on [3].
    Here, there is a soft allignment between the input and output sequence elements.
    A bidirection encoder is used by default. There is no hidden state transfer in this
    model.
    The math:
        Encoder:
        X = Input Sequence of length m.
        H = Bidirection_LSTM(X); Note that here the LSTM has return_sequences = True,
        so H is a sequence of vectors of length m.

        Decoder:
        y(i) = LSTM(s(i-1), y(i-1), v(i)); Where s is the hidden state of the LSTM (h and c)
        and v (called the context vector) is a weighted sum over H:
        v(i) =  sigma(j = 0 to m-1)  alpha(i, j) * H(j)

        The weight alpha[i, j] for each hj is computed as follows:
        energy = a(s(i-1), H(j))
        alhpa = softmax(energy)
        Where a is a feed forward network.
    """

    input_layer
    input = Input(batch_shape=shape)
    input._keras_history[0].supports_masking = True
    encoder = RecurrentContainer(unroll=unroll, stateful=stateful, return_sequences=True, input_length=shape[1])
    encoder.add(LSTMCell(hidden_dim, batch_input_shape=(shape[0], shape[2]), **kwargs))
    for _ in range(1, depth[0]):
        encoder.add(Dropout(dropout))
        encoder.add(LSTMCell(hidden_dim, **kwargs))
    if bidirectional:
        encoder = Bidirectional(encoder, merge_mode='sum')
    encoded = encoder(input)

    decoder = RecurrentContainer(decode=True, output_length=output_length, unroll=unroll, stateful=stateful,
                                 input_length=shape[1])
    decoder.add(Dropout(dropout, batch_input_shape=(shape[0], shape[1], hidden_dim)))
    if depth[1] == 1:
        decoder.add(AttentionDecoderCell(output_dim=output_dim, hidden_dim=hidden_dim))
    else:
        decoder.add(AttentionDecoderCell(output_dim=hidden_dim, hidden_dim=hidden_dim))
        for _ in range(depth[1] - 2):
            decoder.add(Dropout(dropout))
            decoder.add(LSTMDecoderCell(output_dim=hidden_dim, hidden_dim=hidden_dim))
        decoder.add(Dropout(dropout))
        decoder.add(LSTMDecoderCell(output_dim=output_dim, hidden_dim=hidden_dim))
    inputs = [input]
    '''
    if teacher_force:
        truth_tensor = Input(batch_shape=(shape[0], output_length, output_dim))
        inputs += [truth_tensor]
        decoder.set_truth_tensor(truth_tensor)
    '''
    decoded = decoder(encoded)
    model = Model(inputs, decoded)
    return model
