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

class EncoderCNN(nn.Module):
    def __init__(self, in_features, target_size):
        super(EncoderCNN, self).__init__()
        self.linear = nn.Linear(in_features, target_size)
        #self.linear = nn.Linear(resnet.fc.in_features, target_size)
        self.bn = nn.BatchNorm1d(target_size, momentum=0.01)
        self.init_weights()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        for param in self.resnet.parameters():
            param.requires_grad = False

    def get_params(self):
        return list(self.linear.parameters()) + list(self.bn.parameters())

    def init_weights(self):
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)

    def forward(self, images):
        #print("Images shape: ", images.shape, type(images))
        #exit(0)
        features = self.resnet(images)
        features = Variable(features.data)
        features = features.view(features.size(0), -1)
        features = self.linear(features)
        features = self.bn(features)
        return features


class EncoderStory(nn.Module):
    def __init__(self, in_feature_size, img_feature_size, hidden_size, n_layers, embed_size, desc_in_feature_size):
        super(EncoderStory, self).__init__()

        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.cnn = EncoderCNN(in_feature_size, img_feature_size)
        self.lstm = nn.LSTM(img_feature_size, hidden_size, n_layers, batch_first=True, bidirectional=True, dropout=0.5)
        self.linear = nn.Linear(hidden_size * 2 + img_feature_size, hidden_size * 2)
        self.dropout = nn.Dropout(p=0.5)
        self.bn = nn.BatchNorm1d(hidden_size * 2, momentum=0.01)
        self.init_weights()
        self.DescEncoder = nn.LSTM(desc_in_feature_size, desc_hidden_feature_size, 1, batch_first=True, bidirectional=True)
        self.TextEmbed = nn.Embedding(desc_in_feature_size, embed_size)

    def get_params(self):
        return self.cnn.get_params() + list(self.lstm.parameters()) + list(self.linear.parameters()) + list(self.bn.parameters())

    def init_weights(self):
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)

    def forward(self, story_images, storyId, prePicked, AllDescs):
        imgDir = "/home/ashwinsr/resnetFeatMod/"

        if prePicked[0]:
            local_cnn = story_images
            data_size = local_cnn.size()
        else:
            data_size = story_images.size()
            #print(story_images.size())
            #exit(0)
            local_cnn = self.cnn(story_images.view(-1, data_size[2], data_size[3], data_size[4]))
            local_cnn = local_cnn.view(data_size[0], data_size[1], -1)
            for i, id in enumerate(storyId):
                imgPath = imgDir + str(id) + ".npz"
                np.savez(imgPath, local_cnn[i].detach().cpu().numpy())
            #print("Saved a batch of stories: ", storyId[0], " : ", local_cnn[0].shape)
            #exit(0)
        global_rnn, (hn, cn) = self.lstm(local_cnn)#self.lstm(local_cnn.view(data_size[0], data_size[1], -1))
        glocal = torch.cat((local_cnn, global_rnn), 2)
        output = self.linear(glocal)
        output = self.dropout(output)
        output = self.bn(output.contiguous().view(-1, self.hidden_size * 2)).view(data_size[0], data_size[1], -1)

        DescEmbedding = self.TextEmbed(AllDescs)

        return output, (hn, cn), DescEmbedding


class DecoderStory(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab):
        super(DecoderStory, self).__init__()

        self.embed_size = embed_size
        self.linear = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(p=0.5)
        self.rnn = DecoderRNN(embed_size, hidden_size, 2, vocab)
        self.init_weights()

    def get_params(self):
        return list(self.parameters())

    def init_weights(self):
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)

    def forward(self, story_feature, captions, lengths):
        story_feature = self.linear(story_feature)
        story_feature = self.dropout(story_feature)
        story_feature = F.relu(story_feature)
        result = self.rnn(story_feature, captions, lengths)
        return result

    def inference(self, story_feature):
        story_feature = self.linear(story_feature)
        story_feature = F.relu(story_feature)
        result = self.rnn.inference(story_feature)
        return result

class DescAttention(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()

        self.method = method
        self.hidden_size = hidden_size

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size).fill_(1.))

    def forward(self, hidden, encoder_outputs, lens):
        max_len = encoder_outputs.size(1)
        this_batch_size = encoder_outputs.size(0)
        # fix
        # Create variable to store attention energies
        attn_energies = Variable(torch.zeros(this_batch_size, max_len))  # B x S
        attn_energies = attn_energies.to(device)

        # For each batch of encoder outputs

        attn_energies = torch.bmm(encoder_outputs, hidden.unsqueeze(2)).squeeze(2)

        self.state_mask = np.zeros((encoder_outputs.shape[0], encoder_outputs.shape[1]))
        for idx, sl in enumerate(lens):
            self.state_mask[idx, sl:] = 1
        self.state_mask = torch.from_numpy(self.state_mask).type(torch.ByteTensor).to(device)
        attn_energies.masked_fill_(self.state_mask, -float("Inf"))

        # Normalize energies to weights in range 0 to 1, resize to 1 x B x S
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

    def score(self, hidden, encoder_output):

        if self.method == 'dot':
            hidden = hidden.unsqueeze(0)
            encoder_output = encoder_output.unsqueeze(1)
            energy = torch.mm(hidden, encoder_output).item()
            return energy

        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = hidden.dot(energy)
            return energy

        elif self.method == 'concat':
            merged = torch.cat((hidden, encoder_output))
            energy = self.attn(merged)
            energy = self.v.dot(energy)
            return energy

class DescDecoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, batch_size, proj_size, n_layers=1, dropout_p=0.3):
        super(DescDecoder, self).__init__()

        # Define parameters
        # self.cellState = torch.zeros((batch_size, 2*hidden_size)).to(device)
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        # self.max_length = max_length
        self.input_size = input_size
        self.embedding_size = embedding_size
        # Define layers
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.dropout = nn.Dropout(dropout_p)
        self.attn = Attn('dot', hidden_size).to(device)
        self.batch_size = batch_size
        self.lstmcell_1 = nn.LSTMCell(embedding_size + proj_size, hidden_size)
        self.lstmcell_2 = nn.LSTMCell(hidden_size, hidden_size)
        # self.lstmcell = nn.LSTMCell(812, 2 * hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.attn.apply(init_parameters)
        self.embedding.weight.data.normal_(0, 1)
        self.AttentionProjection = nn.Sequential(nn.Linear(hidden_size, proj_size))
        # self.dropout =

    def forward(self, word_input, last_hidden, key, value, cellState, lengths, context):
        # Note that we will only be running forward for a single decoder time step, but will use all encoder outputs
        # Get the embedding of the current input word (last output word)
        word_embedded = self.embedding(word_input.to(device))  # batch x embedding size   .view(1, 1, -1)  # S=1 x B x N
        word_embedded = word_embedded.to(device)

        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat((word_embedded, context), 1)

        rnn_input = rnn_input.to(device)
        #print(rnn_input.shape)
        #print(last_hidden[0].shape)
        #print(cellState[0].shape)
        hidden1, cellState1 = self.lstmcell_1(rnn_input, (last_hidden[0], cellState[0]))
        # cellState1 = cellState.to(device)

        hidden2, cellState2 = self.lstmcell_2(hidden1, (last_hidden[1], cellState[1]))
        # cellState2 = cellState.to(device)

        attention_hidden = self.AttentionProjection(hidden2)

        attn_weights = self.attn(attention_hidden, key, lengths)
        attn_weights = attn_weights.to(device)

        context = attn_weights.bmm(value)  # B x 1 x N
        context = context.squeeze(1)
        context = context.to(device)

        # Final output layer
        output = self.out(hidden2)
        # output = F.log_softmax(self.out(hidden2), dim=1)
        output = output.to(device)
        # del context
        del word_embedded
        # del attn_weights
        # Return final output, hidden state, and attention weights (for visualization)
        return output, (hidden1, hidden2), (cellState1, cellState2), attn_weights, context


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, n_layers, vocab):
        super(DecoderRNN, self).__init__()
        self.vocab = vocab
        vocab_size = len(vocab)
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.dropout1 = nn.Dropout(p=0.1)
        self.lstm = nn.LSTM(embed_size + hidden_size, hidden_size, n_layers, batch_first=True, dropout=0.5)
        self.dropout2 = nn.Dropout(p=0.5)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.softmax = nn.Softmax(0)

        self.brobs = []

        self.init_input = torch.zeros([5, 1, embed_size], dtype=torch.float32)

        if torch.cuda.is_available():
            self.init_input = self.init_input.cuda()

        self.start_vec = torch.zeros([1, vocab_size], dtype=torch.float32)
        self.start_vec[0][1] = 10000
        if torch.cuda.is_available():
            self.start_vec = self.start_vec.cuda()

        self.init_weights()

    def get_params(self):
        return list(self.parameters())

    def init_hidden(self):
        h0 = torch.zeros(1 * self.n_layers, 1, self.hidden_size)
        c0 = torch.zeros(1 * self.n_layers, 1, self.hidden_size)

        h0 = torch.zeros(1 * self.n_layers, 1, self.hidden_size)
        c0 = torch.zeros(1 * self.n_layers, 1, self.hidden_size)
        
        if torch.cuda.is_available():
            h0 = h0.cuda()
            c0 = c0.cuda()
            
        return (h0, c0)

    def init_weights(self):
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0) 

    def forward(self, features, captions, lengths):
        embeddings = self.embed(captions)
        embeddings = self.dropout1(embeddings)
        features = features.unsqueeze(1).expand(-1, np.amax(lengths), -1)
        embeddings = torch.cat((features, embeddings), 2)

        outputs = []
        (hn, cn) = self.init_hidden()

        for i, length in enumerate(lengths):
            lstm_input = embeddings[i][0:length - 1]
            output, (hn, cn) = self.lstm(lstm_input.unsqueeze(0), (hn, cn))
            output = self.dropout2(output)
            output = self.linear(output[0])
            output = torch.cat((self.start_vec, output), 0)
            outputs.append(output)

        return outputs


    def inference(self, features):
        results = []
        (hn, cn) = self.init_hidden()
        vocab = self.vocab
        end_vocab = vocab('<end>')
        forbidden_list = [vocab('<pad>'), vocab('<start>'), vocab('<unk>')]
        termination_list = [vocab('.'), vocab('?'), vocab('!')]
        function_list = [vocab('<end>'), vocab('.'), vocab('?'), vocab('!'), vocab('a'), vocab('an'), vocab('am'), vocab('is'), vocab('was'), vocab('are'), vocab('were'), vocab('do'), vocab('does'), vocab('did')]

        cumulated_word = []
        for feature in features:

            feature = feature.unsqueeze(0).unsqueeze(0)
            predicted = torch.tensor([1], dtype=torch.long).cuda()
            lstm_input = torch.cat((feature, self.embed(predicted).unsqueeze(1)), 2)
            sampled_ids = [predicted,]

            count = 0
            prob_sum = 1.0

            for i in range(50):
                outputs, (hn, cn) = self.lstm(lstm_input, (hn, cn))
                outputs = self.linear(outputs.squeeze(1))

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

                lstm_input = torch.cat((feature, self.embed(predicted).unsqueeze(1)), 2)

            results.append(sampled_ids)

        return results
