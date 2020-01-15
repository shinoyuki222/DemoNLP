# Pytorch NLP - FreeChat

#### To train the model
python pt_seq2seq_atten_train.py

#### To load pretrained model and evaluate 

python pt_seq2seq_atten_train.py -l -cp 4000 -xt

#### Arguments help
python pt_seq2seq_atten_train.py -h








    parser.add_argument('-c','--corpus', action='store', dest='corpus_name', default='core_reduced',help='Store corpus name')

    parser.add_argument('-l','--load', action="store_true",dest='load_model', default=False, help='Load saved model')
    parser.add_argument('-cp','--checkpoint', action="store", dest='checkpoint_iter', default=4000, type=int, help='Set loaded checkpoint_iter')

    parser.add_argument('-xt','--train', action="store_true", dest='skip_train', default=False, help='Skip train model')
    parser.add_argument('-xe','--evaluate', action="store_true", dest='skip_evaluate', default=False, help='Skip evaluate model')

    parser.add_argument('-a','--attn_model', action='store', dest='attn_model', default='dot',help='Store attention mode dot concat or general')

    parser.add_argument('-hs','--hidden_size', action="store", dest='hidden_size', default=500, type=int, help='Set hidden_size')
    parser.add_argument('-en','--encoder_num', action="store", dest='encoder_n_layers', default=2, type=int, help='Set encoder_n_layers')
    parser.add_argument('-dn','--decoder_num', action="store", dest='decoder_n_layers', default=2, type=int, help='Set decoder_n_layers')
    parser.add_argument('-dp','--dropout', action="store", dest='dropout', default=0.1, type=int, help='Set dropout rate')
    parser.add_argument('-b','--batch_size', action="store", dest='batch_size', default=64, type=int, help='Set batch_size')

    parser.add_argument('-n','--n_iteration', action="store", dest='n_iteration', default=4000, type=int, help='Set n_iteration')

    parser.add_argument('-s','--save_every', action="store", dest='save_every', default=500, type=int, help='Set save_every')
    parser.add_argument('-p','--print_every', action="store", dest='print_every', default=1, type=int, help='Set print_every')

