import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models
from torch.autograd import Variable
import torch.nn.functional as F
import itertools
import operator
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Pool
from collections import Counter
import os
import random

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super(MLP, self).__init__()

        layers = list()
        hidden_dim += [output_dim]
        for i in range(len(hidden_dim)):
            layers.append(nn.Linear(input_dim, hidden_dim[i]))
            if i == len(hidden_dim)-1:
                break
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            input_dim = hidden_dim[i]

        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        x = self.fc(x)
        return x


class EncoderCNN(nn.Module):
    '''
    creates feature embedding for images using ResNET-152
    returns B x 196 x 2048 tensor
    '''

    def __init__(self):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        # take the layer that outputs 14x14x2048
        modules = list(resnet.children())[:-3]
        self.resnet = nn.Sequential(*modules)
        for param in self.resnet.parameters():
            param.requires_grad = False

    def get_params(self):
        return list(self.linear.parameters()) + list(self.bn.parameters())

    def init_weights(self):
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)

    def forward(self, images):

        # B x 2048 x D x D
        features = self.resnet(images)
        features = Variable(features.data)
        # B x D x D x 2048
        features = features.permute(0, 2, 3, 1)
        # B x num_pixels x 2048
        features = features.view(features.size(0), -1, features.size(-1))
        return features


class ImageAttention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(ImageAttention, self).__init__()
        # linear layer to transform encoded image
        #self.encoder_att = nn.Sequential(nn.Linear(encoder_dim, attention_dim))
        # linear layer to transform decoder's output
        #self.decoder_att = nn.Sequential(nn.Linear(decoder_dim, attention_dim), nn.Dropout(p=0.1), nn.ReLU())
        self.encoder_key = None
        # softmax layer to calculate weights
        self.softmax = nn.Softmax(dim=1)

    def clear_memory(self):
        self.encoder_key = None

    def forward(self, decoder_hidden, encoder_key, encoder_value):
        # compute this once only, not necessary to do for every time step in decoding
        decoder_state = decoder_hidden
        self.encoder_key = encoder_key
        # compute dot-product attention score i.e energy between current decoder timestep i and all encoder timesteps u
        #print("encoder: ", self.encoder_key.shape)
        #print("decoder: ", decoder_state.shape)
        energy = torch.bmm(self.encoder_key.unsqueeze(0), decoder_state.unsqueeze(-1)).squeeze(dim=-1)
        attention_score = self.softmax(energy)

        # attend encoder value with attention scores
        attended_encoder_state = torch.bmm(attention_score.unsqueeze(1), encoder_value.unsqueeze(0))
        del energy
        del decoder_state

        return attended_encoder_state, attention_score


class DecoderRNN(nn.Module):
    def __init__(self, enc_dim, proj_dim, embed_size, dec_dim, vocab_size, vocab, n_layers=3, dropout_p=0.0):
        super(DecoderRNN, self).__init__()        # Define parameters
        self.vocab = vocab
        self.hidden_size = dec_dim
        self.embed_size = embed_size
        self.output_size = vocab_size
        self.n_layers = n_layers
        # self.dropout_p = dropout_p
        self.proj_dim = proj_dim
        # Define layers
        self.embedding = nn.Embedding(self.output_size, embed_size)
        self.dropout1 = nn.Dropout(p=0.1)
        # self.dropout2 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.2)
        self.attn = ImageAttention(enc_dim, dec_dim, proj_dim)
        self.lstmcells = nn.ModuleList()
        self.lstmcells += [nn.LSTMCell(proj_dim + embed_size, dec_dim).to(device)]
        for layer in range(1, n_layers):
            self.lstmcells += [nn.LSTMCell(dec_dim, dec_dim).to(device)]
        self.out = nn.Linear(dec_dim, vocab_size).to(device)
        self.encoder_key = None
        self.encoder_value = None
        self.state_len = None

        self.start_vec = torch.zeros([1, vocab_size], dtype=torch.float32)
        self.start_vec[0][1] = 10000
        if torch.cuda.is_available():
            self.start_vec = self.start_vec.cuda()

        self.init_h = nn.ParameterList([nn.Parameter(torch.zeros(1, self.hidden_size)) for _ in range(n_layers)])
        self.init_c = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, self.hidden_size)) for _ in range(n_layers)])
        self.init_context = nn.Parameter(torch.zeros(1, self.proj_dim))
        self.softmax = nn.Softmax(0)

    def get_initial_decoder_states(self, context):
        self.attn.clear_memory()

        state = [h.repeat(1, 1).to(context.device) for h in self.init_h]
        cell = [h.repeat(1, 1).to(context.device) for h in self.init_c]
        context = self.init_context.repeat(1, 1).to(context.device)
        return state, cell, context

    def init_encoder_state(self, encoder_key, encoder_value):
        self.encoder_key = encoder_key
        self.encoder_value = encoder_value

    def get_params(self):
        return list(self.parameters())

    def f_next(self, state, cell, context, word_input):
        word_input = word_input.to(device)
        char_embedded = self.embedding(word_input)
        char_embedded = self.dropout1(char_embedded)

        # si = RNN(input=[y_i, c_i-1], hidden=last_hidden)
        #print("Char: ", char_embedded.shape)
        #print("Context: ", context.shape)
        # print("word inp: ", word_input)
        rnn_input = torch.cat((char_embedded, context), -1).to(device)
        state[0], cell[0] = self.lstmcells[0](rnn_input, (state[0], cell[0]))
        for i in range(1, self.n_layers):
            state[i], cell[i] = self.lstmcells[i](state[i-1], ((state[i]), cell[i]))
        last_hidden = state[-1]
        # last_hidden = self.dropout2(state[-1])

        # ci = att( si, h)
        context, attn_scores = self.attn(last_hidden, self.encoder_key, self.encoder_value)
        predicted_char_logits = self.out(last_hidden)

        del word_input
        del char_embedded
        del rnn_input
        del last_hidden
        return state, cell, attn_scores, context.squeeze(0), predicted_char_logits

    def forward_per_sentence(self, keys, values, features, captions, length=0, tf=1.):

        batch_size = 1

        self.init_encoder_state(keys, values)
        # initialize state to last state of encoder, cell & context set to 0
        state, cell, context = self.get_initial_decoder_states(features)

        if captions is not None:
            max_target_len = length
            captions = captions.long()
        else:
            max_target_len = 20

        predicted_seq_logits = torch.zeros(batch_size, max_target_len, self.output_size)
        output_seq = torch.full((batch_size, max_target_len), 0)
        predicted_seq_logits[:, 0, 1] = 10000
        last_word = torch.zeros(1).long()
        last_word[0] = 1

        for t in range(1, max_target_len):
            state, cell, attn_scores, context, predicted_logits = \
                self.f_next(state=state, cell=cell, context=context, word_input=last_word)
            predicted_seq_logits[:, t] = predicted_logits
            predicted_logits = predicted_logits.to(device)
            predicted_word = predicted_logits.max(-1)[1]
            predicted_word = predicted_word.to(device)

            output_seq[:, t] = predicted_word
            # print(last_word.size())
            if random.random() < tf:
                last_word = captions[t].unsqueeze(0)
            else:
                last_word = predicted_word

        return predicted_seq_logits

    def forward(self, keys, values, features, captions, lengths):
        outputs = []
        for i, length in enumerate(lengths):
            # 1 x L x V
            output = self.forward_per_sentence(keys[i], values[i], features[i], captions[i], lengths[i])
            outputs.append(output.squeeze(0))
         # 5 x L x V
        return outputs

    def inference(self, keys, values, features):
        results = []
        #(hn, cn) = self.init_hidden()
        vocab = self.vocab
        end_vocab = vocab('<end>')
        forbidden_list = [vocab('<pad>'), vocab('<start>'), vocab('<unk>')]
        termination_list = [vocab('.'), vocab('?'), vocab('!')]
        function_list = [vocab('<end>'), vocab('.'), vocab('?'), vocab('!'), vocab('a'), vocab('an'), vocab('am'),
                         vocab('is'), vocab('was'), vocab('are'), vocab('were'), vocab('do'), vocab('does'),
                         vocab('did')]

        cumulated_word = []
        for idx, feature in enumerate(features):
            feature = feature.unsqueeze(0)
            predicted = torch.tensor([1], dtype=torch.long).cuda()
            #lstm_input = torch.cat((feature, self.embed(predicted).unsqueeze(1)), 2)
            sampled_ids = [predicted, ]

            self.init_encoder_state(keys[idx], values[idx])#, text_key[idx], text_val[idx], desc_len[idx])
            # initialize state to last state of encoder, cell & context set to 0
            state, cell, context = self.get_initial_decoder_states(feature)

            count = 0
            prob_sum = 1.0

            for i in range(50):
                #  L x V
                state, cell, _, context, outputs = self.f_next(state=state, cell=cell, context=context, word_input=predicted)
                #outputs = outputs.squeeze(0)
                if predicted not in termination_list:
                    outputs[0][end_vocab] = -100.0

                for forbidden in forbidden_list:
                    outputs[0][forbidden] = -100.0

                cumulated_counter = Counter()
                cumulated_counter.update(cumulated_word)

                prob_res = outputs[0]
                prob_res = self.softmax(prob_res)
                for word, cnt in cumulated_counter.items():
                    if cnt > 0 and word not in function_list:
                        prob_res[word] = prob_res[word] / (1.0 + cnt * 5.0)
                prob_res = prob_res * (1.0 / prob_res.sum())

                candidate = []
                for i in range(100):
                    index = np.random.choice(prob_res.size()[0], 1, p=prob_res.cpu().detach().numpy())[0]
                    candidate.append(index)

                counter = Counter()
                counter.update(candidate)

                sorted_candidate = sorted(counter.items(), key=operator.itemgetter(1), reverse=True)

                predicted, _ = counter.most_common(1)[0]
                cumulated_word.append(predicted)

                predicted = torch.from_numpy(np.array([predicted])).cuda()
                sampled_ids.append(predicted)

                if predicted == 2:
                    break

                #lstm_input = torch.cat((feature, self.embed(predicted).unsqueeze(1)), 2)

            results.append(sampled_ids)

        return results


class EncoderStory(nn.Module):
    def __init__(self, in_feature_size, img_feature_size, hidden_size, n_layers, proj_dim):
        super(EncoderStory, self).__init__()
        self.img_dir = "/home/ashwinsr/resnetFeatMod/"
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.cnn = EncoderCNN()
        self.key_projector = MLP(input_dim=in_feature_size, output_dim=proj_dim, hidden_dim=[])
        self.value_projector = MLP(input_dim=in_feature_size, output_dim=proj_dim, hidden_dim=[])
        #self.init_weights()

    def get_params(self):
        return self.cnn.get_params() + list(self.linear.parameters()) + list(self.bn.parameters())

    def init_weights(self):
        self.key_projector.weight.data.normal_(0.0, 0.02)
        self.value_projector.bias.data.fill_(0)

    def forward(self, story_images, storyId, prePicked):
        if prePicked[0]:
            # B x 5 x num_pixel x 2048
            local_cnn = story_images
        else:
            data_size = story_images.size()
            # (B x 5) x numpixels x 2048
            local_cnn = self.cnn(story_images.view(-1, data_size[2], data_size[3], data_size[4]))
            # B x 5 x numpixels x 2048
            local_cnn = local_cnn.view(data_size[0], data_size[1], -1, local_cnn.size(-1))
            #print(local_cnn.shape)
            for i, id in enumerate(storyId):
                imgPath = self.img_dir + str(id) + ".npz"
                np.savez(imgPath, local_cnn[i].detach().cpu().numpy())

        key = self.key_projector(local_cnn)
        value = self.value_projector(local_cnn)
        return key, value, local_cnn


class DecoderStory(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab, enc_dim, proj_dim):
        super(DecoderStory, self).__init__()

        self.embed_size = embed_size
        self.linear = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(p=0.5)
        self.rnn = DecoderRNN(dec_dim=hidden_size, embed_size=embed_size, vocab_size=len(vocab),
                              proj_dim=proj_dim, n_layers=2, enc_dim=enc_dim, vocab=vocab)
        self.init_weights()

    def get_params(self):
        return list(self.parameters())

    def init_weights(self):
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)

    def forward(self, keys, values, story_feature, captions, lengths):
        result = self.rnn(keys, values, story_feature, captions, lengths)
        return result

    def inference(self, keys, values, story_feature):
        result = self.rnn.inference(keys, values, story_feature)
        return result
