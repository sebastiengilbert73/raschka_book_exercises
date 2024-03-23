# Cf. p. 508
import logging
import torch

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s %(message)s')

def main():
    logging.info("create_rnn.main()")

    torch.manual_seed(1)

    rnn_layer = torch.nn.RNN(
        input_size=5, hidden_size=2, num_layers=1, batch_first=True
    )
    w_xh = rnn_layer.weight_ih_l0
    w_hh = rnn_layer.weight_hh_l0
    b_xh = rnn_layer.bias_ih_l0
    b_hh = rnn_layer.bias_hh_l0
    logging.info(f"w_xh.shape = {w_xh.shape}")
    logging.info(f"w_hh.shape = {w_hh.shape}")
    logging.info(f"b_xh.shape = {b_xh.shape}")
    logging.info(f"b_hh.shape = {b_hh.shape}")

    x_seq = torch.tensor([[1.0]*5, [2.0]*5, [3.0]*5]).float()  # (3, 5)  sequence_length=3, num_features=5
    logging.info(f"x_seq.shape = {x_seq.shape}")
    # Output of the simple RNN
    output, hn = rnn_layer(torch.reshape(x_seq, (1, 3, 5)))  # B=1, sequence_length=3, num_features=5
    # Manually computing the output
    out_man = []
    for t in range(3):
        xt = torch.reshape(x_seq[t], (1, 5))  # (1, 5)
        logging.info(f"Time step {t} =>")
        logging.info(f"   Input         : {xt.numpy()}")
        ht = torch.matmul(xt, torch.transpose(w_xh, 0, 1)) + b_xh
        logging.info(f"   Hidden        : {ht.detach().numpy()}")
        if t > 0:
            prev_h = out_man[t - 1]
        else:
            prev_h = torch.zeros(ht.shape)
        ot = ht + torch.matmul(prev_h, torch.transpose(w_hh, 0, 1)) + b_hh
        ot = torch.tanh(ot)
        out_man.append(ot)
        logging.info(f"   Output (manual) : {ot.detach().numpy()}")
        logging.info(f"   RNN output      : {output[:, t].detach().numpy()}")

if __name__ == '__main__':
    main()