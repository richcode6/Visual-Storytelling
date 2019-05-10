import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
import random
from data_loader import get_loader
from build_vocab import Vocabulary
from model_att import EncoderStory, DecoderStory
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image
# nltk.download('punkt')

import time

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def predict(encoder, decoder, curr_key, curr_value, feature, text_key_curr,
            text_val_curr, desc_len_curr, captions, vocab, text_feat_curr):
    encoder.eval()
    decoder.eval()
    sentences = []
    target_sentences = []
    inference_results = decoder.inference(curr_key, curr_value, feature, text_key_curr,
                                          text_val_curr, desc_len_curr, text_feat_curr)
    for i, result in enumerate(inference_results):
        temp = []
        for cap in captions[i]:
            temp.append(vocab.idx2word[cap.item()])
        newTemp = []
        for w in temp:
            if w == '<end>' or w == '<pad>' or w == '<start>':
                continue

            newTemp.append(w)
        target_sentences.append(newTemp)
        words = []
        target_words = []
        for word_id in result:
            word = vocab.idx2word[word_id.item()]
            words.append(word)
            if word == '<end>':
                break

        try:
            words.remove('<start>')
        except Exception:
            pass

        try:
            words.remove('<end>')
        except Exception:
            pass
        sentences.append(' '.join(words))

    result = {}
    result["story_text_normalized"] = sentences[0] + " " + sentences[1] + " " + sentences[2] + " " + \
                                      sentences[3] + " " + sentences[4]

    print("Predicted: " + result["story_text_normalized"])
    print("---------")
    print("Target: " + ' '.join(target_sentences[0]) + ' '.join(target_sentences[1]) + ' '.join(
        target_sentences[2]) + ' '.join(target_sentences[3]) + \
          ' '.join(target_sentences[4]))



def main(args):
    # Create model directory
    Log_file = open("Perp_logger.txt", "a")
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # Image preprocessing
    train_transform = transforms.Compose([
        transforms.RandomCrop(args.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    val_transform = transforms.Compose([
        transforms.Resize(args.image_size, interpolation=Image.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    # Load vocabulary wrapper.
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Build data loader
    train_data_loader, cnn_in_feature = get_loader(args.train_image_dir, args.train_sis_path, args.train_dii_path, vocab, train_transform, args.batch_size,
                                                   "/data/VOL4/ashwinsr/GLACNet/story_ID_list_train.pkl", shuffle=True, num_workers=args.num_workers, isVal=False)
    val_data_loader, cnn_in_feature = get_loader(args.val_image_dir, args.val_sis_path, args.val_dii_path, vocab, val_transform, args.batch_size,
                                                 "/data/VOL4/ashwinsr/GLACNet/story_ID_list_val.pkl", shuffle=False, num_workers=args.num_workers, isVal=True)
    #train_data_loader=val_data_loader

    encoder = EncoderStory(cnn_in_feature, len(vocab), args.embed_size, args.hidden_size, args.num_layers, args.proj_dim)
    decoder = DecoderStory(args.embed_size,args.hidden_size, len(vocab), args.hidden_size, args.proj_dim, vocab)
    #print(encoder)
    #print("-------------------------------------------------------", "Hierarchical model")
    #print(decoder)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=args.learning_rate * 5,
                                         weight_decay=args.weight_decay)
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(decoder_optimizer, patience=2, threshold=1e-4)
    pretrained_epoch = 0

    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()
        print("Cuda is enabled...")

    if args.pretrained_epoch > 0:
        print("pretrained loaded")
        pretrained_epoch = args.pretrained_epoch
        path = os.path.join(args.model_path,"model_state_"+str(args.pretrained_epoch))
        loaded = torch.load(path)
        encoder.load_state_dict(loaded['encoder_state_dict'])
        decoder.load_state_dict(loaded['decoder_state_dict'])
        encoder_optimizer.load_state_dict(loaded['encoder_opt_state_dict'])
        decoder_optimizer.load_state_dict(loaded['decoder_opt_state_dict'])
        scheduler.load_state_dict(loaded['scheduler_state_dict'])

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    total_train_step = len(train_data_loader)
    total_val_step = len(val_data_loader)

    min_avg_loss = float("inf")
    overfit_warn = 0
    start = time.time()
    for epoch in range(args.num_epochs):

        if epoch < pretrained_epoch:
            continue

        encoder.train()
        decoder.train()
        avg_loss = 0.0
        for bi, (image_stories, targets_set, lengths_set, photo_squence_set, album_ids_set, storyID, isPreLoaded, AllDescs, desc_length_set) in enumerate(train_data_loader):
            decoder.zero_grad()
            encoder.zero_grad()
            loss = 0
            images = to_var(torch.stack(image_stories))

            # B x 5 x dim
            keys, values, features, text_key, text_val, text_feat = encoder(images, storyID, isPreLoaded, AllDescs)

            for si, data in enumerate(zip(features, targets_set, lengths_set)):
                feature = data[0]
                captions = to_var(data[1])
                lengths = data[2]
                # 5 x L x vocab_dim
                outputs = decoder(keys=keys[si], values=values[si], story_feature=feature,
                                  captions=captions, lengths=lengths, text_key=text_key[si],
                                  text_val=text_val[si], desc_len=desc_length_set[si], text_feat=text_feat[si])
                # print(outputs[0])
                # print(captions[0])
                # print(lengths[0])

                for sj, result in enumerate(zip(outputs, captions, lengths)):
                    # print(result[0])
                    # print(result[1][0:result[2]])
                    loss += criterion(result[0], result[1][0:result[2]].detach().cpu())
                    # print(loss.item())
                # exit()

            avg_loss += loss.item()
            loss /= (args.batch_size * 5)
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()

            # Print log info
            if bi % args.log_step == 0:
                print('Epoch [%d/%d], Train Step [%d/%d], Loss: %.4f, Perplexity: %5.4f, Time: %.4f'
                      %(epoch + 1, args.num_epochs, bi, total_train_step,
                        loss.item(), np.exp(loss.item()), time.time() - start))

        avg_loss /= (args.batch_size * total_train_step * 5)
        print('Epoch [%d/%d], Average Train Loss: %.4f, Average Train Perplexity: %5.4f' %(epoch + 1, args.num_epochs, avg_loss, np.exp(avg_loss)) + "/n")
        Log_file.write(
            "Epoch " + str(epoch + 1) + " Average Train Loss: " + str(avg_loss) + " Average Train Perplexity: " + str(
                np.exp(avg_loss)) + "\n")
        # Save the models
        name = os.path.join(args.model_path,"model_state_"+str(epoch+1))

        torch.save({
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'encoder_opt_state_dict': encoder_optimizer.state_dict(),
            'decoder_opt_state_dict': decoder_optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch': epoch
        }, name)

        # Validation
        encoder.eval()
        decoder.eval()
        avg_loss = 0.0
        epoch = 0
        for bi, (image_stories, targets_set, lengths_set, photo_sequence_set, album_ids_set, storyID, isPreLoaded, AllDescs, desc_length_set) in enumerate(val_data_loader):
            loss = 0
            images = to_var(torch.stack(image_stories))

            keys, values, features, text_key, text_val, text_feat = encoder(images, storyID, isPreLoaded, AllDescs)

            for si, data in enumerate(zip(features, targets_set, lengths_set)):
                feature = data[0]
                captions = to_var(data[1])
                lengths = data[2]
                outputs = decoder(keys=keys[si], values=values[si], story_feature=feature,
                                  captions=captions, lengths=lengths, text_key=text_key[si],
                                  text_val=text_val[si], desc_len=desc_length_set[si], text_feat=text_feat[si])

                for sj, result in enumerate(zip(outputs, captions, lengths)):
                    loss += criterion(result[0], result[1][0:result[2]].detach().cpu())

                if bi == 10 and si == 3:
                    print("For story :::: ", storyID[0], bi, si)
                    predict(encoder=encoder, decoder=decoder, curr_key=keys[si], curr_value=values[si],
                            text_val_curr=text_val[si], text_key_curr=text_key[i], desc_len_curr=desc_len[si],
                            feature=feature, captions=captions, vocab=vocab, text_feat_curr=text_feat[si])

            avg_loss += loss.item()
            loss /= (args.batch_size * 5)

            # Print log info
            if bi % args.log_step == 0:
                print('Epoch [%d/%d], Val Step [%d/%d], Loss: %.4f, Perplexity: %5.4f'
                      %(epoch + 1, args.num_epochs, bi, total_val_step,
                        loss.item(), np.exp(loss.item())))

        avg_loss /= (args.batch_size * total_val_step * 5)
        print('Epoch [%d/%d], Average Val Loss: %.4f, Average Val Perplexity: %5.4f' %(epoch + 1, args.num_epochs, avg_loss, np.exp(avg_loss)))
        Log_file.write(
            "Epoch VAL" + str(epoch + 1) + " Average val Loss: " + str(avg_loss) + " Average val Perplexity: " + str(
                np.exp(avg_loss)))
        scheduler.step(avg_loss)
        #Termination Condition
        overfit_warn = overfit_warn + 1 if (min_avg_loss < avg_loss) else 0
        min_avg_loss = min(min_avg_loss, avg_loss)

        if overfit_warn >= 5:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/project/ocean/home/rrnigam/vist/hierAttModels' ,
                        help='path for saving trained models')
    parser.add_argument('--image_size', type=int, default=224 ,
                        help='size for input images')
    parser.add_argument('--vocab_path', type=str, default='/project/ocean/home/ashwinsr/glacExp/hierAttModels/models/vocab.pkl',
                        help='path for vocabulary wrapper')
    parser.add_argument('--train_image_dir', type=str, default='/project/ocean/home/rrnigam/vist/images/train-curated/' ,
                        help='directory for resized train images')
    parser.add_argument('--val_image_dir', type=str, default='/project/ocean/home/rrnigam/vist/images/val-curated/' ,
                        help='directory for resized val images')
    parser.add_argument('--train_sis_path', type=str,
                        default='/project/ocean/home/rrnigam/vist/sis/train.story-in-sequence.json',
                        help='path for train sis json file')
    parser.add_argument('--val_sis_path', type=str,
                        default='/project/ocean/home/rrnigam/vist/sis/val.story-in-sequence.json',
                        help='path for val sis json file')
    parser.add_argument('--log_step', type=int , default=10,
                        help='step size for prining log info')
    parser.add_argument('--img_feature_size', type=int , default=1024 ,
                        help='dimension of image feature')
    parser.add_argument('--embed_size', type=int , default=256 ,
                        help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512 ,
                        help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=2 ,
                        help='number of layers in lstm')
    parser.add_argument('--pretrained_epoch', type=int, default=0)
    parser.add_argument('--num_epochs', type=int, default=75)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--learning_rate', type=float, default=0.0015)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--proj_dim', type=int, default=512)

    parser.add_argument('--train_dii_path', type=str,
                        default='/project/ocean/home/rrnigam/vist/dii/train.description-in-isolation.json',
                        help='path for train sis json file')
    parser.add_argument('--val_dii_path', type=str,
                        default='/project/ocean/home/rrnigam/vist/dii/val.description-in-isolation.json',
                        help='path for val sis json file')
    args = parser.parse_args()
    print(args)
    main(args)
