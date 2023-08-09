import tensorflow as tf
from spektral.layers import GATConv


class ST_GAT(tf.keras.Model):

    def __init__(self, gat_features=12, out_channels=9, n_nodes=228, heads=8, dropout=0.0):
        """
        Initialize the ST-GAT model
        :param in_channels Number of input channels
        :param out_channels Number of output channels
        :param n_nodes Number of nodes in the graph
        :param heads Number of attention heads to use in graph
        :param dropout Dropout probability on output of Graph Attention Network
        """
        super(ST_GAT, self).__init__()
        self.n_pred = out_channels
        self.heads = heads
        self.dropout = dropout
        self.n_nodes = n_nodes

        self.n_preds = 9
        lstm1_hidden_size = 32
        lstm2_hidden_size = 128

        # single graph attentional layer with 8 attention heads
        self.gat = GATConv(channels=gat_features, attn_heads=heads,
                           concat_heads=False, dropout_rate=dropout)

        self.lstm1 = tf.keras.layers.LSTM(units=lstm1_hidden_size, return_sequences=True, return_state=True)
        self.lstm2 = tf.keras.layers.LSTM(units=lstm2_hidden_size, return_sequences=True, return_state=True)

        self.dense1 = tf.keras.layers.Dense(self.n_nodes * self.n_preds)

    def call(self, inputs):
        x, a = inputs
        # x = tf.random.normal([50, 228, 12])
        # a = tf.random.normal([50, 228, 228])
        gat_1 = self.gat([x, a])
        # print(gat_1.shape)

        batch_size = 50
        n_node = 228
        gat_1 = tf.reshape(gat_1, (batch_size, n_node, 12))
        gat_1 = tf.transpose(gat_1, perm=[2, 0, 1])

        # print('before lstm 1:', gat_1.shape)

        lstm_out, _, _ = self.lstm1(gat_1)

        # print('before lstm 2:', lstm_out.shape)

        lstm2_out, _, _ = self.lstm2(lstm_out)

        # print('after lstm 2:', lstm2_out.shape)
        lstm2_out = tf.squeeze(lstm2_out[-1:, :, :])
        dense_out = self.dense1(lstm2_out)

        # print('after dense :', dense_out.shape)

        # Now reshape into final output
        s = dense_out.shape
        # [50, 228*9] -> [50, 228, 9]
        dense_out = tf.reshape(dense_out, (s[0], n_node, self.n_pred))
        # [50, 228, 9] ->  [11400, 9]
        dense_out = tf.reshape(dense_out, (s[0] * n_node, self.n_pred))
        # print('last : ', dense_out.shape)
        return dense_out

    def model(self):
        x = tf.keras.Input(shape=(50, 228, 12))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))
