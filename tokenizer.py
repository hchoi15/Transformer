import argparse
import sentencepiece as spm
import logging

LOGGER = logging.getLogger()
#getLogger(): 이름이 제공되는 경우 지정된 이름을 가진 로거 인스턴스에 대한 참조를 반환하고, 그렇지 않으면 root를 반환

#parsing: 일련의 문자열을 의미있는 토큰으로 분해하고 이들로 이루어진 파스 트리를 만드는 과정
def get_args():
    parser = argparse.ArgumentParser()
    #ArgumentParser(): ArgumentParser 객체를 생성
    #add_argument(): 프로그램 인자에 대한 정보를 채움 (일반적으로 명령행의 문자열을 객체로 변환하는 방법을 알려줌)
    #parse_args(): 인자를 파싱, 명령행을 검사하고 각 인자를 적절한 형으로 변환한 다음 적절한 액션을 호출
    
    # Run settings
    parser.add_argument('--vocab_size', default=16000, type=int) #이름, default=인자가 명령행에 없는 경우 생성되는 값, type=명령행 인자가 변환되어야 하는 형
    parser.add_argument('--datapath', default='./datasets/iwslt17.fr.en')
    parser.add_argument('--src_path', default='./datasets/iwslt17.fr.en/train.fr-en.en')
    parser.add_argument('--dest_path', default='./datasets/iwslt17.fr.en/train.fr-en.fr')
    parser.add_argument('--langpair', default='fr-en')

    args = parser.parse_args() #인자 없이 호출
    return args

class WordpieceTokenizer(object):
    def __init__(self, datapath, vocab_size=0, l=0, alpha=0, n=0):
        #info(): 프로그램의 정상 작동 중에 발생하는 이벤트 보고
        logging.info("vocab_size={}".format(vocab_size))
        self.templates = '--input={} --model_prefix={} --vocab_size={} --bos_id=2 --eos_id=3 --pad_id=0 --unk_id=1'
        self.vocab_size = vocab_size
        self.spm_path = "{}/sp".format(datapath)

        # for subword regualarization
        self.l = l
        self.alpha = alpha
        self.n = n
        
    def transform(self, sentence, max_length=0):
        if self.l and self.alpha:
            x = self.sp.SampleEncodeAsIds(sentence, self.l, self.alpha)
        elif self.n:
            x = self.sp.NBestEncodeAsIds(sentence, self.n)
        else:
            x = self.sp.EncodeAsIds(sentence)
        if max_length>0:
            pad = [0]*max_length
            pad[:min(len(x),max_length)] = x[:min(len(x),max_length)]
            x = pad
        return x
    
    def fit(self, input_file):
        cmd = self.templates.format(input_file, self.spm_path, self.vocab_size, 0)
        spm.SentencePieceTrainer.Train(cmd)
        
    def load_model(self):
        file = self.spm_path + ".model"
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(file)
        self.sp.SetEncodeExtraOptions('eos')
        print("load_model {}".format(file))
        return self

    def decode(self,encoded_sentences):
        decoded_output = []
        for encoded_sentence in encoded_sentences:
            x = self.sp.DecodeIds(encoded_sentence)
            decoded_output.append(x)
        return decoded_output

    def __len__(self):
        return len(self.sp)

def main():
    """
    Train SentencePiece Model
    """
    args = get_args() #def get_args
    tokenizer = WordpieceTokenizer(datapath=args.datapath, vocab_size=args.vocab_size) #class WordpieceTokenizer
    tokenizer.fit(",".join([args.src_path,args.dest_path])) #class WordpieceTokenizer, def fit
    
if __name__ == "__main__":
    main()