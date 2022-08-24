from wordsegment import load, segment
load()

import PyPDF2
import os,sys
from tqdm import tqdm
from pandas import Series
import pickle
import shutil

sys.setrecursionlimit(5000)
print(sys.getrecursionlimit())


def get_abstract(text, seg=True):
    if 'abstract' in text and 'keyword' in text and 'ntroduction' in text:
        ai, ki, ii = text.index('abstract'),  text.index('keyword'), text.index('ntroduction')
        if ai < ki < ii:
            text = text.split('abstract')[1].split('keyword')[0].strip().replace('\n', ' ')
        elif ai < ii < ki:
            text = text.split('abstract')[1].split('ntroduction')[0].strip().replace('\n', ' ')[:-1]
        elif ai < ki:
            text = text.split('abstract')[1].split('keyword')[0].strip().replace('\n', ' ')
        elif ai < ii:
            text = text.split('abstract')[1].split('ntroduction')[0].strip().replace('\n', ' ')[:-1]

    elif 'abstract' in text and 'keyword' in text:
        text = text.split('abstract')[1].split('keyword')[0].strip().replace('\n', ' ')

    elif 'abstract' in text and 'ntroduction' in text:
        text = text.split('abstract')[1].split('ntroduction')[0].strip().replace('\n', ' ')[:-1]

    elif 'abstract' not in text:
        if 'keyword' in text:
            text = text.split('keyword')[0]
        elif 'ntroduction' in text:
            text = text.split('ntroduction')[0][:-1]
    else:
        return None

    if seg:
        text = ' '.join(segment(text))

    return text


def pdf2abstract(pdf_file):
    pdfFileObj = open(pdf_file , 'rb')
    pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
    text = pdfReader.getPage(0).extractText().lower() + \
           pdfReader.getPage(1).extractText().lower()
    
    abstract = get_abstract(text)
    return abstract


def predictPdfFileTask(pdf_file, model, tfidf):
    # abstract = pdf2abstract('/home/thesun/Books/papers/Attention Is All You Need.pdf')
    abstract = pdf2abstract(pdf_file)
    if abstract:
        encoded_abstract = tfidf.transform(Series(abstract)).toarray()
        task = model.predict(encoded_abstract)[0]
        return task
    return 'Other'



model = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))


pdf_files = os.listdir('/home/thesun/Books/papers')
cv_files  = os.listdir('/home/thesun/Books/organized_papers/Computer vision/')
nlp_files = os.listdir('/home/thesun/Books/organized_papers/NLP/')
ssl_files = os.listdir('/home/thesun/Books/organized_papers/Self supervised learning/')
rl_files  = os.listdir('/home/thesun/Books/organized_papers/Reinforcement leanring/')
oprganized_files = cv_files + nlp_files + ssl_files + rl_files

for file in tqdm(pdf_files):
    if file not in oprganized_files:
        task = predictPdfFileTask(f'/home/thesun/Books/papers/{file}', model, tfidf)
        if   task == 'Computer Vision':
            shutil.copy2(f'/home/thesun/Books/papers/{file}', f'/home/thesun/Books/organized_papers/Computer vision/{file}')
        elif task == 'NLP':
            shutil.copy2(f'/home/thesun/Books/papers/{file}', f'/home/thesun/Books/organized_papers/NLP/{file}')
        elif task == 'Contrastive learning':
            shutil.copy2(f'/home/thesun/Books/papers/{file}', f'/home/thesun/Books/organized_papers/Self supervised learning/{file}')
        elif task == 'Reinforcement learning':
            shutil.copy2(f'/home/thesun/Books/papers/{file}', f'/home/thesun/Books/organized_papers/Reinforcement leanring/{file}')
        else:
            shutil.copy2(f'/home/thesun/Books/papers/{file}', f'/home/thesun/Books/organized_papers/Other/{file}')